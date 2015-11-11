package chris;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

import moa.classifiers.meta.BayesianBagAdaptiveNXj2;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomTreeGenerator;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;
import weka.core.Utils;

public class TestGammaNXj2 {
	
	public static void main(String[] args) {
		
		ArrayList<Integer> tokens = new ArrayList<Integer>( 200 + 100 + 500 );
		
		BayesianBagAdaptiveNXj2 bag = new BayesianBagAdaptiveNXj2();
		
		//HyperplaneGenerator gen = new HyperplaneGenerator();
		RandomRBFGenerator gen = new RandomRBFGenerator();
		gen.numCentroidsOption.setValue(50);
		int nc = 40;
		gen.numClassesOption.setValue(nc);
		gen.prepareForUse();
		bag.ensembleSizeOption.setValue(1);
		bag.prepareForUse();
		bag.setDebug(true);
		double m_instCounts = 1;
		double[] classDist = new double[nc];
		for(int x= 0; x < classDist.length; x++) classDist[x] = 1;
		
		for(int x = 0; x < 10000; x++) {
			//bag.trainOnInstance( hpg10.nextInstance() );
			//tokens.add( (int) rtg40.nextInstance().classValue() );
			
			Instance inst = gen.nextInstance();
			int classValue = (int)inst.classValue();
			GammaDistribution g = new GammaDistribution( classDist[classValue],
					m_instCounts / (classDist[classValue]*classDist[classValue]) );
			System.out.println(g.sample());
			
			m_instCounts += 1;
			classDist[ classValue ] += 1;
			
		}
		
		//System.err.println( hpg10.getHeader() );
		
		
		System.exit(0);

		Collections.shuffle(tokens);
		
		//double[] classDist = new double[] { 200, 100, 500 };
		//double N = Utils.sum(classDist);

		ArrayList<Double> arr = new ArrayList<Double>();
		double instCount = 1;
		classDist = new double[40];
		for(int x = 0; x < 40; x++) classDist[x] = 1;
		
		Random rnd = new Random(0);
		
		for(int x = 0; x < tokens.size(); x++) {
			int classIndex = tokens.get(x);
			for(int b = 0; b < 1; b++) {
				GammaDistribution g = new GammaDistribution(classDist[classIndex],
						instCount / (classDist[classIndex]*classDist[classIndex]) );
				g.reseedRandomGenerator(rnd.nextLong());
				arr.add( g.sample() );
			}		
			instCount += 1;
			classDist[classIndex] += 1;
		}
		double[] samples = new double[ arr.size() ];
		for(int x = 0; x < samples.length; x++) samples[x] = arr.get(x);

		
		System.out.println("theoretical e[w]: " + classDist.length);
		System.out.println("empirical e[w]: " + Utils.mean(samples) );
		
		// -----
		
		// what about xj, for j=1
		
		/*
		
		samples = new double[10000];
		GammaDistribution g = new GammaDistribution(classDist[0], N / (classDist[0]*classDist[0]));
		for(int x = 0; x < 10000; x++) {
			samples[x] = g.sample();
		}
		System.out.println("theoretical e[w1]: " + (N / classDist[0]));
		System.out.println("empirical e[w1]: " + Utils.mean(samples));
		
		*/
		
	}

}
