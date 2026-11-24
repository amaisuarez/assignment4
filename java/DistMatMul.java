import com.hazelcast.client.HazelcastClient;
import com.hazelcast.cluster.Member;
import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.IExecutorService;
import com.hazelcast.map.IMap;

import java.io.IOException;
import java.lang.management.ManagementFactory;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.concurrent.Future;

/**
 * Driver of the distributed multiplication (a Hazelcast client, not a member: it
 * stores no data and runs no tasks). One process = one configuration
 * (n, members m, worker threads per member t) on an already running cluster.
 *
 * usage: DistMatMul --cluster NAME --members h:p,h:p,... --expect-members M --threads-per-member T
 *                   --n N --a A.bin --b B.bin [--warmup W] [--reps R] [--out C.bin]
 *
 * Partitioning (row blocks, "1-D" decomposition):
 *   - C and A are cut into P = m * t row blocks (sizes differ by at most one row,
 *     the same blockBounds as Assignment 3). Block b is placed on member b mod m,
 *     so every member gets t blocks = one per executor thread.
 *   - B is cut into m row panels, one stored on each member. Every task needs all
 *     of B, so each member fetches the m - 1 panels it does not own once per
 *     multiplication (see BlockTask).
 *   Hazelcast decides where a key lives by hashing it to one of 271 partitions. To
 *   control placement, the driver searches for keys ("A3#17", "B1#5", ...) whose
 *   partition is owned by the wanted member. The search is done once, before timing.
 *
 * One multiplication = three phases, timed by the driver with System.nanoTime():
 *   scatter  cut A into blocks and B into panels (array copies) and putAll them
 *            into the IMaps A and B (driver -> members)
 *   compute  submit the P BlockTasks with submitToKeyOwner and wait for all of
 *            them. Includes the redistribution of B between members, the kernel,
 *            and storing the C blocks in IMap C (on the member that computed them)
 *   gather   getAll of the C blocks (members -> driver) and copy them into C
 *   total = scatter + compute + gather. The phases run one after the other (the
 *   driver waits for each one), so they do not overlap and can be added. Inside
 *   compute the tasks overlap with each other, so their own timings are reported
 *   per task and are NOT summed into the phase time.
 *   Not timed: reading the input files, starting the members, connecting the
 *   client, the key search, and the cleanup after each multiplication (clearing
 *   the three maps and the members' B caches).
 *
 * Output (stdout, CSV-like; '#' lines are metadata):
 *   iter,kind,idx,total_ns,scatter_ns,compute_ns,gather_ns,checksum
 *   task,kind,idx,block,member,thread,rows,a_local,fetched_b,read_a_ns,get_b_ns,fetch_b_ns,kernel_ns,put_c_ns,remote_b_bytes
 *   memstat,<MemberOps.Stats fields>
 */
public final class DistMatMul {

    static double[] loadDense(String path, int n) throws IOException {
        ByteBuffer buf = ByteBuffer.wrap(Files.readAllBytes(Path.of(path))).order(ByteOrder.LITTLE_ENDIAN);
        if (buf.capacity() != 8L * n * n)
            throw new IOException(path + ": expected " + (long) n * n + " float64 values");
        double[] m = new double[n * n];
        buf.asDoubleBuffer().get(m);
        return m;
    }

    static void saveDense(String path, double[] C) throws IOException {
        ByteBuffer buf = ByteBuffer.allocate(8 * C.length).order(ByteOrder.LITTLE_ENDIAN);
        buf.asDoubleBuffer().put(C);
        Files.write(Path.of(path), buf.array());
    }

    static double sum(double[] x) {
        double s = 0.0;
        for (double v : x) s += v;
        return s;
    }

    /** First key "prefix#salt" whose partition is owned by `owner`. */
    static String keyOwnedBy(HazelcastInstance client, String prefix, Member owner) {
        for (int salt = 0; salt < 1_000_000; salt++) {
            String k = prefix + "#" + salt;
            Member o = client.getPartitionService().getPartition(k).getOwner();
            if (o != null && o.getUuid().equals(owner.getUuid())) return k;
        }
        throw new IllegalStateException("no key found for member " + owner);
    }

    final HazelcastInstance client;
    final IMap<String, double[]> mapA, mapB, mapC;
    final IExecutorService exec;
    final int n, m, t, blocks;
    final int[] aLo, bLo;
    final String[] aKeys, bKeys;
    long runCounter = System.nanoTime();     // unique id per multiplication (B cache key on the members)
    final List<BlockTask.Stats> lastStats = new ArrayList<>();

    DistMatMul(HazelcastInstance client, int n, int t) {
        this.client = client; this.n = n; this.t = t;
        mapA = client.getMap(HzSetup.MAP_A);
        mapB = client.getMap(HzSetup.MAP_B);
        mapC = client.getMap(HzSetup.MAP_C);
        exec = client.getExecutorService(HzSetup.EXECUTOR);
        List<Member> members = new ArrayList<>(client.getCluster().getMembers());
        m = members.size();
        blocks = m * t;
        aLo = Parallel.blockBounds(n, blocks);
        bLo = Parallel.blockBounds(n, m);
        aKeys = new String[blocks];
        for (int b = 0; b < blocks; b++) aKeys[b] = keyOwnedBy(client, "A" + b, members.get(b % m));
        bKeys = new String[m];
        for (int i = 0; i < m; i++) bKeys[i] = keyOwnedBy(client, "B" + i, members.get(i));
    }

    /** One multiplication C = A * B; returns {scatter_ns, compute_ns, gather_ns}. */
    long[] multiply(double[] A, double[] B, double[] C) throws Exception {
        long run = ++runCounter;
        lastStats.clear();

        long t0 = System.nanoTime();
        Map<String, double[]> partsA = new HashMap<>();
        for (int b = 0; b < blocks; b++) partsA.put(aKeys[b], Arrays.copyOfRange(A, aLo[b] * n, aLo[b + 1] * n));
        Map<String, double[]> partsB = new HashMap<>();
        for (int i = 0; i < m; i++) partsB.put(bKeys[i], Arrays.copyOfRange(B, bLo[i] * n, bLo[i + 1] * n));
        mapA.putAll(partsA);
        mapB.putAll(partsB);
        long t1 = System.nanoTime();

        int[] bLoPanels = Arrays.copyOf(bLo, m);
        List<Future<BlockTask.Stats>> fs = new ArrayList<>(blocks);
        for (int b = 0; b < blocks; b++) {
            int rows = aLo[b + 1] - aLo[b];
            fs.add(exec.submitToKeyOwner(new BlockTask(run, b, aKeys[b], rows, n, bKeys, bLoPanels), aKeys[b]));
        }
        for (Future<BlockTask.Stats> f : fs) lastStats.add(f.get());
        long t2 = System.nanoTime();

        Map<String, double[]> partsC = mapC.getAll(new HashSet<>(Arrays.asList(aKeys)));
        for (int b = 0; b < blocks; b++) {
            double[] c = partsC.get(aKeys[b]);
            System.arraycopy(c, 0, C, aLo[b] * n, c.length);
        }
        long t3 = System.nanoTime();
        return new long[]{t1 - t0, t2 - t1, t3 - t2};
    }

    /** Untimed: empty the maps and the members' B caches before the next multiplication. */
    void cleanup() throws Exception {
        mapA.clear();
        mapB.clear();
        mapC.clear();
        for (Future<Integer> f : exec.submitToAllMembers(new MemberOps.ClearCache()).values()) f.get();
    }

    static void waitForCluster(HazelcastInstance client, int expect) throws Exception {
        long deadline = System.nanoTime() + 120_000_000_000L;
        while (client.getCluster().getMembers().size() < expect) {
            if (System.nanoTime() > deadline) throw new IllegalStateException("cluster did not reach " + expect + " members");
            Thread.sleep(100);
        }
        if (client.getCluster().getMembers().size() != expect)
            throw new IllegalStateException("cluster has " + client.getCluster().getMembers().size()
                    + " members, expected " + expect);
        // First partition operation assigns the partitions; then wait until no migration is running.
        IMap<String, double[]> probe = client.getMap(HzSetup.MAP_A);
        probe.set("probe", new double[0]);
        probe.delete("probe");
        IExecutorService exec = client.getExecutorService(HzSetup.EXECUTOR);
        while (true) {
            boolean safe = true;
            for (Future<Boolean> f : exec.submitToAllMembers(new MemberOps.ClusterSafe()).values()) safe &= f.get();
            if (safe) break;
            if (System.nanoTime() > deadline) throw new IllegalStateException("cluster not safe");
            Thread.sleep(100);
        }
    }

    public static void main(String[] args) throws Exception {
        String cluster = null, members = null, pa = null, pb = null, out = null;
        int n = -1, expect = -1, t = 1, warmup = 0, reps = 1;
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--cluster" -> cluster = args[i + 1];
                case "--members" -> members = args[i + 1];
                case "--expect-members" -> expect = Integer.parseInt(args[i + 1]);
                case "--threads-per-member" -> t = Integer.parseInt(args[i + 1]);
                case "--n" -> n = Integer.parseInt(args[i + 1]);
                case "--a" -> pa = args[i + 1];
                case "--b" -> pb = args[i + 1];
                case "--warmup" -> warmup = Integer.parseInt(args[i + 1]);
                case "--reps" -> reps = Integer.parseInt(args[i + 1]);
                case "--out" -> out = args[i + 1];
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (cluster == null || members == null || expect <= 0 || n <= 0 || pa == null || pb == null) {
            System.err.println("usage: DistMatMul --cluster NAME --members h:p,... --expect-members M "
                    + "--threads-per-member T --n N --a A.bin --b B.bin [--warmup W] [--reps R] [--out C.bin]");
            System.exit(1);
        }
        if ((long) n * n > Integer.MAX_VALUE) throw new IllegalArgumentException("n too large for a flat double[]");
        HzSetup.quietLogging();

        double[] A = loadDense(pa, n), B = loadDense(pb, n);
        double[] C = new double[n * n];

        long c0 = System.nanoTime();
        HazelcastInstance client = HazelcastClient.newHazelcastClient(
                HzSetup.client(cluster, Arrays.asList(members.split(","))));
        try {
            waitForCluster(client, expect);
            long c1 = System.nanoTime();
            DistMatMul dm = new DistMatMul(client, n, t);
            long c2 = System.nanoTime();
            System.out.println("#cluster_members," + dm.m);
            System.out.println("#threads_per_member," + t);
            System.out.println("#blocks," + dm.blocks);
            System.out.println("#connect_ns," + (c1 - c0));
            System.out.println("#key_placement_ns," + (c2 - c1));
            System.out.println("#scatter_payload_bytes," + 16L * n * n);
            System.out.println("#gather_payload_bytes," + 8L * n * n);
            System.out.println("#b_redistribution_payload_bytes," + 8L * n * n * (dm.m - 1));

            for (int r = 0; r < warmup + reps; r++) {
                boolean measured = r >= warmup;
                String kind = measured ? "measured" : "warmup";
                int idx = measured ? r - warmup : r;
                Arrays.fill(C, Double.NaN);          // a missing block would show up as NaN
                long[] ph = dm.multiply(A, B, C);
                System.out.println("iter," + kind + "," + idx + "," + (ph[0] + ph[1] + ph[2]) + ","
                        + ph[0] + "," + ph[1] + "," + ph[2] + "," + sum(C));
                for (BlockTask.Stats s : dm.lastStats) {
                    System.out.println("task," + kind + "," + idx + "," + s.block() + "," + s.member() + ","
                            + s.thread() + "," + s.rows() + "," + s.aLocal() + "," + s.fetchedB() + ","
                            + s.readANs() + "," + s.getBNs() + "," + s.fetchBNs() + "," + s.kernelNs() + ","
                            + s.putCNs() + "," + s.remoteBBytes());
                }
                if (r == warmup + reps - 1) {        // memory of every member with A, B, C still stored
                    for (Future<String> f : dm.exec.submitToAllMembers(new MemberOps.Stats()).values())
                        System.out.println("memstat," + f.get());
                }
                dm.cleanup();
            }
        } finally {
            client.shutdown();
        }
        if (out != null) saveDense(out, C);
        Runtime rt = Runtime.getRuntime();
        System.out.println("#driver_heap_used_bytes," + (rt.totalMemory() - rt.freeMemory()));
        System.out.println("#driver_peak_thread_count," + ManagementFactory.getThreadMXBean().getPeakThreadCount());
    }
}
