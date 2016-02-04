package chris;

import moa.streams.generators.RandomRBFGeneratorNR;

public class CreateTypicalRBF {
	
	public static void main(String[] args) throws Exception {
	
		RandomRBFGeneratorNR gen = new RandomRBFGeneratorNR();
		gen.numClassesOption.setValue(5);
		gen.numCentroidsOption.setValue(10);
		gen.prepareForUse();
		System.out.println(gen.getHeader());
		for(int x = 0; x < 100000; x++) {
			System.out.println(gen.nextInstance());
		}
	
	}

}
