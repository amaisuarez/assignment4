import com.hazelcast.core.HazelcastInstance;
import com.hazelcast.core.HazelcastInstanceAware;
import com.hazelcast.map.LocalMapStats;

import java.io.Serial;
import java.io.Serializable;
import java.lang.management.ManagementFactory;
import java.util.concurrent.Callable;

/** Small untimed housekeeping tasks that the driver runs on the members. */
final class MemberOps {

    private MemberOps() {}

    /** True when no partition migration is in progress (all partitions have their owner). */
    static final class ClusterSafe implements Callable<Boolean>, Serializable, HazelcastInstanceAware {
        @Serial private static final long serialVersionUID = 1L;
        private transient HazelcastInstance hz;
        @Override public void setHazelcastInstance(HazelcastInstance hz) { this.hz = hz; }
        @Override public Boolean call() { return hz.getPartitionService().isClusterSafe(); }
    }

    /** Drops the member's cached copy of B (see BlockTask). */
    static final class ClearCache implements Callable<Integer>, Serializable {
        @Serial private static final long serialVersionUID = 1L;
        @Override public Integer call() { return BlockTask.clearCache(); }
    }

    /**
     * Memory and thread figures of one member, as one CSV fragment:
     * pid,address,processors,heap_used_now,heap_used_after_gc,heap_max,
     * entry_cost_A,entry_cost_B,entry_cost_C,live_threads,peak_threads
     * entry_cost_* = Hazelcast's LocalMapStats.getOwnedEntryMemoryCost(): bytes used
     * by the entries of that map stored on this member (serialized form).
     */
    static final class Stats implements Callable<String>, Serializable, HazelcastInstanceAware {
        @Serial private static final long serialVersionUID = 1L;
        private transient HazelcastInstance hz;
        @Override public void setHazelcastInstance(HazelcastInstance hz) { this.hz = hz; }

        @Override public String call() {
            Runtime rt = Runtime.getRuntime();
            long usedNow = rt.totalMemory() - rt.freeMemory();
            System.gc();
            long usedAfterGc = rt.totalMemory() - rt.freeMemory();
            StringBuilder sb = new StringBuilder();
            sb.append(ProcessHandle.current().pid()).append(',')
              .append(hz.getCluster().getLocalMember().getAddress()).append(',')
              .append(rt.availableProcessors()).append(',')
              .append(usedNow).append(',').append(usedAfterGc).append(',').append(rt.maxMemory());
            for (String m : new String[]{HzSetup.MAP_A, HzSetup.MAP_B, HzSetup.MAP_C}) {
                LocalMapStats s = hz.getMap(m).getLocalMapStats();
                sb.append(',').append(s.getOwnedEntryMemoryCost());
            }
            var th = ManagementFactory.getThreadMXBean();
            sb.append(',').append(th.getThreadCount()).append(',').append(th.getPeakThreadCount());
            return sb.toString();
        }
    }
}
