package chris;

import java.io.File;
import java.io.PrintStream;

import moa.streams.generators.RandomTreeGenerator;

public class GenerateRTGClassDrift {
	
	public static void main(String[] args) throws Exception {
		
		if(args.length == 0) {
			args = new String[] { "80", "7", "100000", "10" };
		    //System.setOut(new PrintStream(new File("/tmp/nothing.txt")));
		}
		
		int k = Integer.parseInt(args[0]);
		int d = Integer.parseInt(args[1]);
		int maxInstances = Integer.parseInt(args[2]);
		int maxDrifts = Integer.parseInt(args[3]);
		
		RandomTreeGenerator rtg = new RandomTreeGenerator();
		rtg.numClassesOption.setValue(k);
		rtg.maxTreeDepthOption.setValue(d);
		rtg.prepareForUse();
		
		ClassDistributionDriftGenerator gen = new ClassDistributionDriftGenerator(rtg, maxInstances, maxDrifts);
		gen.generate();
		
	}

}
