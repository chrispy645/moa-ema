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
		int[] numClasses = new int[] { 5, 10, 20, 40 };
		
		for(int numClass : numClasses) {
		
			
			RandomTreeGenerator rtg = new RandomTreeGenerator();
			rtg.numClassesOption.setValue(numClass);
			rtg.prepareForUse();
			m_classDist = new double[ rtg.getHeader().numClasses() ];
			for(int x = 0; x < 10000; x++) m_classDist[ (int) rtg.nextInstance().classValue() ] += 1;
			System.out.println("rtg_" + numClass + ": " + Arrays.toString(m_classDist));
			
			
			/*
			RandomRBFGenerator rbf = new RandomRBFGenerator();
			rbf.numClassesOption.setValue(numClass);
			rbf.numCentroidsOption.setValue(10);
			rbf.prepareForUse();
			m_classDist = new double[ rbf.getHeader().numClasses() ];
			for(int x = 0; x < 1000000; x++) {
				Instance inst = rbf.nextInstance();
				//System.out.println(inst.classValue());
				m_classDist[ (int) inst.classValue() ] += 1;
			}
			System.out.println("rbf_" + numClass + ": " + Arrays.toString(m_classDist));
			
			
			RandomRBFGenerator rbf2 = new RandomRBFGenerator();
			rbf2.numClassesOption.setValue(numClass);
			rbf2.numCentroidsOption.setValue(50);
			rbf2.prepareForUse();	
			m_classDist = new double[ rbf2.getHeader().numClasses() ];
			for(int x = 0; x < 10000; x++) m_classDist[ (int) rbf2.nextInstance().classValue() ] += 1;
			System.out.println("rbf2_" + numClass + ": " + Arrays.toString(m_classDist));
			*/
			
			RandomRBFGenerator rbf2k = new RandomRBFGenerator();
			rbf2k.numClassesOption.setValue(numClass);
			rbf2k.numCentroidsOption.setValue(numClass*2);
			rbf2k.prepareForUse();
			m_classDist = new double[ rbf2k.getHeader().numClasses() ];
			for(int x = 0; x < 1000000; x++) {
				Instance inst = rbf2k.nextInstance();
				//System.out.println(inst.classValue());
				m_classDist[ (int) inst.classValue() ] += 1;
			}
			System.out.println("rbf2k_" + numClass + ": " + Arrays.toString(m_classDist));			
			
			HyperplaneGenerator hpg = new HyperplaneGenerator();
			hpg.numClassesOption.setValue(numClass);
			hpg.prepareForUse();
			m_classDist = new double[ hpg.getHeader().numClasses() ];
			for(int x = 0; x < 10000; x++) m_classDist[ (int) hpg.nextInstance().classValue() ] += 1;
			System.out.println("hpg_" + numClass + ": " + Arrays.toString(m_classDist));	
		
		}
		
		ArffFileStream covType = new ArffFileStream();
		covType.arffFileOption.setValue("/Users/cjb60/github/590-cyber-sec/data/covtypeNorm.arff");
		covType.prepareForUse();
		m_classDist = new double[ covType.getHeader().numClasses() ];
		for(int x = 0; x < 100000; x++) m_classDist[ (int) covType.nextInstance().classValue() ] += 1;
		System.out.println("covtype: " + Arrays.toString(m_classDist));	
		
		ArffFileStream elec = new ArffFileStream();
		elec.arffFileOption.setValue("/Users/cjb60/github/590-cyber-sec/data/elecNormNew.arff");
		elec.prepareForUse();
		m_classDist = new double[ elec.getHeader().numClasses() ];
		while(elec.hasMoreInstances()) m_classDist[ (int) elec.nextInstance().classValue() ] += 1;
		System.out.println("elec: " + Arrays.toString(m_classDist));	
		
		ArffFileStream poker = new ArffFileStream();
		poker.arffFileOption.setValue("/Users/cjb60/github/590-cyber-sec/data/poker-lsn.arff");
		poker.prepareForUse();
		m_classDist = new double[ poker.getHeader().numClasses() ];
		for(int x = 0; x < 100000; x++) m_classDist[ (int) poker.nextInstance().classValue() ] += 1;
		System.out.println("poker: " + Arrays.toString(m_classDist));	
		
		/*
		
		// sample from gammas to verify theory
		double k = 10; // assume 10
		double xj = 100;
		double n = 1000;
		
		GammaDistribution kxj = new GammaDistribution(xj, k / xj);
		double[] samples = new double[1000];
		for(int x = 0; x < samples.length; x++) samples[x] = kxj.sample();
		System.out.println("empirical mean: " + Utils.mean(samples));
		System.out.println("empirical variance: " + Utils.variance(samples));
		System.out.println("theoretical mean: " + k);
		System.out.println("theoretical variance: " + (k*k) / xj);
		
		GammaDistribution nxj2 = new GammaDistribution(xj, n / (xj*xj));
		samples = new double[1000];
		for(int x = 0; x < samples.length; x++) samples[x] = nxj2.sample();
		System.out.println("empirical mean: " + Utils.mean(samples));
		System.out.println("empirical variance: " + Utils.variance(samples));
		System.out.println("theoretical mean: " + n / xj);
		System.out.println("theoretical variance: " + ((n*n) / (xj*xj*xj)) );
		
		*/
		
		
	}

}
