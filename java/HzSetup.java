import com.hazelcast.client.config.ClientConfig;
import com.hazelcast.config.Config;
import com.hazelcast.config.JoinConfig;
import com.hazelcast.config.NetworkConfig;

import java.util.List;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 * Hazelcast configuration shared by the cluster members (ClusterMember.java) and the
 * driver (DistMatMul.java, a Hazelcast client). Everything that affects the
 * measurements is set here explicitly instead of relying on defaults.
 *
 *  - Discovery: TCP/IP member list only (multicast and auto-detection off), so a
 *    member never joins a cluster that is not part of the experiment. The cluster
 *    name is unique per run as a second guard.
 *  - Maps A, B and C: backup-count 0 and async-backup-count 0. Each value is stored
 *    exactly once, so memory and transfer volumes are not doubled by replicas. (A
 *    production deployment would keep backups for fault tolerance; this is a
 *    documented choice of the study.) In-memory format: BINARY (default), i.e. the
 *    serialized double[].
 *  - Executor "matmul": pool size = worker threads per member (t). Unbounded queue.
 *  - Phone-home and client statistics disabled; Hazelcast logging limited to WARNING.
 */
final class HzSetup {

    static final String MAP_A = "A", MAP_B = "B", MAP_C = "C", EXECUTOR = "matmul";

    /** Strong reference, otherwise the logger (and its level) can be garbage collected. */
    private static final Logger HZ_LOG = Logger.getLogger("com.hazelcast");

    private HzSetup() {}

    static void quietLogging() {
        HZ_LOG.setLevel(Level.WARNING);
    }

    static Config member(String cluster, String host, int port, List<String> memberList, int execThreads) {
        Config c = new Config();
        c.setClusterName(cluster);
        c.setProperty("hazelcast.phone.home.enabled", "false");
        c.setProperty("hazelcast.logging.type", "jdk");
        c.setProperty("hazelcast.local.localAddress", host);

        NetworkConfig net = c.getNetworkConfig();
        net.setPort(port).setPortAutoIncrement(false);
        JoinConfig join = net.getJoin();
        join.getMulticastConfig().setEnabled(false);
        join.getAutoDetectionConfig().setEnabled(false);
        join.getTcpIpConfig().setEnabled(true).setMembers(memberList);

        c.getExecutorConfig(EXECUTOR).setPoolSize(execThreads).setQueueCapacity(0).setStatisticsEnabled(true);
        for (String m : List.of(MAP_A, MAP_B, MAP_C))
            c.getMapConfig(m).setBackupCount(0).setAsyncBackupCount(0).setStatisticsEnabled(true);
        return c;
    }

    static ClientConfig client(String cluster, List<String> memberList) {
        ClientConfig cc = new ClientConfig();
        cc.setClusterName(cluster);
        cc.setProperty("hazelcast.logging.type", "jdk");
        cc.setProperty("hazelcast.client.statistics.enabled", "false");
        cc.getNetworkConfig().setAddresses(memberList);          // smart routing (default): talks to every member
        cc.getConnectionStrategyConfig().getConnectionRetryConfig().setClusterConnectTimeoutMillis(60_000);
        return cc;
    }
}
