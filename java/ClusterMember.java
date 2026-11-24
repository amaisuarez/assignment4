import com.hazelcast.core.Hazelcast;
import com.hazelcast.core.HazelcastInstance;

import java.util.Arrays;
import java.util.List;
import java.util.concurrent.CountDownLatch;

/**
 * One Hazelcast cluster member = one worker process (one JVM).
 *
 * usage: ClusterMember --cluster NAME --host 127.0.0.1 --port 5701 --threads T
 *               --members 127.0.0.1:5701,127.0.0.1:5702,...
 *
 * --threads is the size of the member's "matmul" executor pool, i.e. how many
 * block tasks this member can run at the same time. When the member has joined
 * the cluster it prints "#member_ready,<pid>,<address>" on stdout (the benchmark
 * runner waits for that line) and then runs until the process is terminated.
 *
 * To use several physical machines, start one or more members on every machine
 * with --host set to that machine's address and the same --members list.
 */
public final class ClusterMember {

    public static void main(String[] args) throws InterruptedException {
        String cluster = null, host = "127.0.0.1", members = null;
        int port = 5701, threads = 1;
        for (int i = 0; i + 1 < args.length; i += 2) {
            switch (args[i]) {
                case "--cluster" -> cluster = args[i + 1];
                case "--host" -> host = args[i + 1];
                case "--port" -> port = Integer.parseInt(args[i + 1]);
                case "--threads" -> threads = Integer.parseInt(args[i + 1]);
                case "--members" -> members = args[i + 1];
                default -> throw new IllegalArgumentException("unknown option " + args[i]);
            }
        }
        if (cluster == null || members == null || threads <= 0) {
            System.err.println("usage: ClusterMember --cluster NAME [--host H] [--port P] --threads T --members h:p,h:p,...");
            System.exit(1);
        }
        HzSetup.quietLogging();
        List<String> list = Arrays.asList(members.split(","));
        HazelcastInstance hz = Hazelcast.newHazelcastInstance(HzSetup.member(cluster, host, port, list, threads));
        System.out.println("#member_ready," + ProcessHandle.current().pid() + ","
                + hz.getCluster().getLocalMember().getAddress());
        System.out.flush();
        new CountDownLatch(1).await();      // serve tasks until the process is terminated
    }
}
