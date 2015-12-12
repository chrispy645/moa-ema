package chris;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import weka.core.Instance;
import weka.core.Utils;
import moa.core.MiscUtils;
import moa.streams.ArffFileStream;
import moa.streams.InstanceStream;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomTreeGenerator;

public class ExamineClassDist {
	
	public static void main(String[] args) {
	
		double[] m_classDist = null;
		int[] numClasses = new int[] { 80, 160 };

		int[] numzerodist = new int[6];
	
		for(int z = 0; z < numClasses.length; z++) {
			int numClass = numClasses[z];
			RandomTreeGenerator rtg = new RandomTreeGenerator();
			rtg.numClassesOption.setValue(numClass);
			//rtg.treeRandomSeedOption.setValue(seed);
			rtg.maxTreeDepthOption.setValue(7);
			rtg.prepareForUse();
			m_classDist = new double[ rtg.getHeader().numClasses() ];
			for(int x = 0; x < 100000; x++) m_classDist[ (int) rtg.nextInstance().classValue() ] += 1;

			System.out.println(Arrays.toString(m_classDist));

			int numzero = 0;
			for(int x = 0; x < m_classDist.length; x++) {
				if(m_classDist[x] == 0) {
					numzero++;
				}
			}		
			numzerodist[z] = numzero;	
		}
		
		System.out.println(Arrays.toString(numzerodist));
		
	}

}
