import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.map.IMap;

import java.io.Serial;
import java.io.Serializable;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.Callable;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ConcurrentHashMap;

/**
 * The unit of distributed work: computes one row block of C on the member that
 * owns that block of A.
 *
 * The driver submits the task with submitToKeyOwner(task, key), where `key` is
 * the IMap key of the A block. Hazelcast therefore runs it on the member that
 * stores the block, so reading A is a local operation, and C is written under
 * the same key, so it is stored on the same member (a local write).
 *
 * Every task needs the whole of B. B is stored in the cluster as one row panel
 * per member. The first task of a member that arrives in a given multiplication
 * (`run`) fetches all panels with one getAll (the panels of the other members
 * travel over the network) and assembles them into one n x n array; the other
 * tasks of the same member and run wait for that array and reuse it. So B is
 * transferred once per member, not once per task. The cache is cleared by
 * MemberOps.ClearCache after every multiplication (untimed).
 *
 * Kernel: Kernels.ikjRows (the Assignment 3 row kernel, itself the Assignment 2
 * i-k-j kernel) applied to the local block: rows 0..rows-1 of the block of A
 * times the full B. For every C[i][j] the products are added in the same order
 * k = 0..n-1 as the sequential kernel, so the result is bit-identical to it.
 *
 * The task returns its own timings (System.nanoTime() of the member's JVM; only
 * durations are reported, clocks of different processes are never compared).
 */
final class BlockTask implements Callable<BlockTask.Stats>, Serializable, HazelcastInstanceAware {

    @Serial private static final long serialVersionUID = 1L;

    /** Timings and placement of one task, sent back to the driver. */
    record Stats(String member, String thread, int block, int rows, boolean aLocal, boolean fetchedB,
                 long readANs, long getBNs, long fetchBNs, long kernelNs, long putCNs,
                 long remoteBBytes) implements Serializable {}

    /** B assembled on this member for one run, plus who paid for fetching it. */
    private record FetchedB(double[] b, long fetchNs, long remoteBytes) {}

    private static final ConcurrentHashMap<Long, CompletableFuture<FetchedB>> B_CACHE = new ConcurrentHashMap<>();

    private final long run;
    private final int block, rows, n;
    private final String key;
    private final String[] bKeys;
    private final int[] bLo;
    private transient HazelcastInstance hz;

    BlockTask(long run, int block, String key, int rows, int n, String[] bKeys, int[] bLo) {
        this.run = run; this.block = block; this.key = key; this.rows = rows; this.n = n;
        this.bKeys = bKeys; this.bLo = bLo;
    }

    @Override public void setHazelcastInstance(HazelcastInstance hz) { this.hz = hz; }

    static int clearCache() {
        int size = B_CACHE.size();
        B_CACHE.clear();
        return size;
    }

    private boolean isLocal(String k) {
        return hz.getPartitionService().getPartition(k).getOwner().localMember();
    }

    @Override public Stats call() {
        long t0 = System.nanoTime();
        IMap<String, double[]> mapA = hz.getMap(HzSetup.MAP_A);
        double[] a = mapA.get(key);
        if (a == null || a.length != rows * n) throw new IllegalStateException("block " + key + " missing or wrong size");
        long t1 = System.nanoTime();

        CompletableFuture<FetchedB> mine = new CompletableFuture<>();
        CompletableFuture<FetchedB> f = B_CACHE.putIfAbsent(run, mine);
        boolean fetcher = f == null;
        if (fetcher) {
            f = mine;
            try {
                mine.complete(fetchB());
            } catch (RuntimeException e) {
                mine.completeExceptionally(e);
                throw e;
            }
        }
        FetchedB fb = f.join();
        long t2 = System.nanoTime();

        double[] c = new double[rows * n];
        Kernels.ikjRows(a, fb.b(), c, n, 0, rows);
        long t3 = System.nanoTime();

        IMap<String, double[]> mapC = hz.getMap(HzSetup.MAP_C);
        mapC.set(key, c);
        long t4 = System.nanoTime();

        return new Stats(hz.getCluster().getLocalMember().getAddress().toString(), Thread.currentThread().getName(),
                block, rows, isLocal(key), fetcher, t1 - t0, t2 - t1, fetcher ? fb.fetchNs() : 0, t3 - t2, t4 - t3,
                fetcher ? fb.remoteBytes() : 0);
    }

    private FetchedB fetchB() {
        long t0 = System.nanoTime();
        IMap<String, double[]> mapB = hz.getMap(HzSetup.MAP_B);
        Map<String, double[]> parts = mapB.getAll(Set.of(bKeys));
        double[] b = new double[n * n];
        long remote = 0;
        for (int i = 0; i < bKeys.length; i++) {
            double[] p = parts.get(bKeys[i]);
            if (p == null) throw new IllegalStateException("B panel " + bKeys[i] + " missing");
            System.arraycopy(p, 0, b, bLo[i] * n, p.length);
            if (!isLocal(bKeys[i])) remote += 8L * p.length;
        }
        return new FetchedB(b, System.nanoTime() - t0, remote);
    }
}
