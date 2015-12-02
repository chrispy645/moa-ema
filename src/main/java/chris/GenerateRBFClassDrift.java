package chris;

import moa.streams.generators.RandomRBFGeneratorNR;
import moa.streams.generators.RandomTreeGenerator;

public class GenerateRBFClassDrift {
	
	public static void main(String[] args) {
		
		if(args.length == 0) {
			args = new String[] { "80", "7", "100000", "10" };
		}
		
		int k = Integer.parseInt(args[0]);
		int n = Integer.parseInt(args[1]); // num centroids total
		int maxInstances = Integer.parseInt(args[2]);
		int maxDrifts = Integer.parseInt(args[3]);
		
		RandomRBFGeneratorNR rbf = new RandomRBFGeneratorNR();
		rbf.numClassesOption.setValue(k);
		rbf.numCentroidsOption.setValue(n);
		rbf.prepareForUse();
		
		ClassDistributionDriftGeneratorOld gen = new ClassDistributionDriftGeneratorOld(rbf, maxInstances, maxDrifts);
		gen.generate();
		
	}

}
