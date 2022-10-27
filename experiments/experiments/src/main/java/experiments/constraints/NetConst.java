package experiments.constraints;

import java.util.Arrays;
import java.util.Optional;

interface NetConst {
	boolean test(int tcDelay, int tcRate);

	record ExactNetConst(int tcDelay, int tcRate) implements NetConst {
		public boolean test(int tcDelay, int tcRate) {
			return this.tcDelay == tcDelay && this.tcRate == tcRate;
		}
	};

	record AnyDelayExactRate(int tcRate) implements NetConst {
		public boolean test(int tcDelay, int tcRate) {
			return this.tcRate == tcRate;
		}
	};

	record AnyRateExactDelay(int tcDelay) implements NetConst {
		public boolean test(int tcDelay, int tcRate) {
			return this.tcDelay == tcDelay;
		}
	};


	class NetConstWhitelist implements NetConst {

		private final Iterable<NetConst> netConstsWhitelist;

		public NetConstWhitelist(Iterable<NetConst> netConstsWhitelist) {
			this.netConstsWhitelist = netConstsWhitelist;
		}

		public final static NetConst[] NET_CONSTRAINTS_WHITELIST = {
			new ExactNetConst(-1, -1),
	//		new NetConst(Optional.empty(), Optional.of(-2)),
		};

		@Override public boolean test(int tcDelay, int tcRate) {
			for(NetConst netConst : this.netConstsWhitelist) {
				if(netConst.test(tcDelay, tcRate)) {
					return true;
				}
			}
			return false;
		}
	}

	public static boolean TEST_DISABLED_TC(Optional<Integer> tc_delay, Optional<Integer> tc_rate) {
		if(tc_delay.isPresent() && tc_rate.isPresent()) {
			int tcDelay = tc_delay.get();
			int tcRate = tc_rate.get();
			return new NetConst.ExactNetConst(-1, -1).test(tcDelay, tcRate);
		}
		return true;
	}

	public static boolean TEST_STD_CONSTRAINTS(Optional<Integer> tc_delay, Optional<Integer> tc_rate) {
		if(tc_delay.isPresent() && tc_rate.isPresent()) {
			int tcDelay = tc_delay.get();
			int tcRate = tc_rate.get();
			NetConst netConst = new NetConst.NetConstWhitelist(Arrays.asList(
					new NetConst.ExactNetConst(-1, -1),
					new NetConst.AnyDelayExactRate(-1),
					new NetConst.AnyRateExactDelay(-1),
					// tc_delay= -1, 1, 2, 5, 15, 50
					// tc_rate= -1, 10000, 1000, 100, 10, 1
					new NetConst.ExactNetConst(5, 10000),
					new NetConst.ExactNetConst(15, 100),
					new NetConst.ExactNetConst(50, 1)
			));
			if(!netConst.test(tcDelay, tcRate)) {
				return false;
			}
		}
		return true;
	}

}

