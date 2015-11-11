package moa.streams.generators;

import java.util.Arrays;

import weka.core.Instance;

public class TestRandomRBFGeneratorNR {
	
	public static void main(String[] args) {
		
		RandomRBFGeneratorNR rbf = new RandomRBFGeneratorNR();
		rbf.numClassesOption.setValue(40);
		rbf.numCentroidsOption.setValue(40);
		rbf.prepareForUse();
		int[] classDist = new int[ rbf.getHeader().numClasses() ];
		for(int x = 0; x < 10000; x++) {
			Instance inst = rbf.nextInstance();
			classDist[ (int) inst.classValue() ] += 1;
		}
		System.out.println(Arrays.toString(classDist));
		
	}

}
