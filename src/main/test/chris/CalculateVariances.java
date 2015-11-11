package chris;

import java.util.Arrays;
import java.util.Random;

import org.apache.commons.math3.distribution.GammaDistribution;
import org.apache.commons.math3.random.JDKRandomGenerator;
import org.apache.commons.math3.random.RandomGenerator;

import weka.core.Instance;
import weka.core.Utils;
import moa.streams.ArffFileStream;
import moa.streams.InstanceStream;
import moa.streams.generators.HyperplaneGenerator;
import moa.streams.generators.RandomRBFGenerator;
import moa.streams.generators.RandomTreeGenerator;

public class CalculateVariances {
	
	private static RandomGenerator rg = new JDKRandomGenerator();
	static {
		rg.setSeed(0);
	}
	
	static class ExpVarPair {
		public double expectation;
		public double variance;
		public ExpVarPair(double expectation, double variance) {
			this.expectation = expectation;
			this.variance = variance;
		}
		public String toString() {
			return "expectation = " + this.expectation + ", var = " + this.variance;
		}
	}
	
	static class ThreePair {
		public double[] dist;
		public double[] kxjs;
		public double[] nxj2s;
		public ThreePair(double[] dist, double[] kxjs, double[] nxj2s) {
			this.dist = dist;
			this.kxjs = kxjs;
			this.nxj2s = nxj2s;
		}
	}
	
	private static ThreePair runAndGetDistribution(InstanceStream g, int N) {
		double[] dist = new double[ g.getHeader().numClasses() ];
		double k = g.getHeader().numClasses();
		for(int j = 0; j < dist.length; j++) dist[j] = 1; // laplace
		double[] kxj_samples = new double[N];
		double[] nxj2_samples = new double[N];
		for(int x = 1; x <= N; x++) {
			Instance inst = g.nextInstance();
			
			/*
			GammaDistribution kxj = new GammaDistribution(
					dist[(int)inst.classValue()],
					k / dist[(int)inst.classValue()] 
			);
			GammaDistribution nxj2 = new GammaDistribution(
					dist[(int)inst.classValue()],
					(double)x / (dist[(int)inst.classValue()]*dist[(int)inst.classValue()]) 
			);
			
			kxj_samples[x-1] = kxj.sample();
			nxj2_samples[x-1] = nxj2.sample();
			*/
			
			dist[ (int) inst.classValue() ] += 1;
		}
		return new ThreePair(dist, null, null);
	}
	
	public static ExpVarPair getExpectationAndVarianceForGammaKXj(double[] dist) {
		double N = Utils.sum(dist);
		double k = dist.length;
		
		return new ExpVarPair(k, Math.pow(k, 3) / N );
	}
	
	public static ExpVarPair getExpectationAndVarianceForGammaNXj2(double[] dist) {
		double N = Utils.sum(dist);
		int k = dist.length;
		double sum = 0;
		for(int j = 0; j < k; j++) {
			
			double f1 = 1.0 / (dist[j]*dist[j]);
			double f2 = 1.0 / dist[j];
			sum += (N * (f1 + f2));
		}
		sum = sum - (k*k);
		return new ExpVarPair(k, sum);
	}
	
	public static void main(String[] args) {
		
		// define artificial datasets
		
		System.out.println("dataset,nxj2_mean,nxj2_var,kxj_mean,kxj_var");
		
		int numClasses[] = new int[] { 5, 10, 20, 40 };
		
		int N = 100000;
		
		for(int numClass : numClasses) {
		
			RandomTreeGenerator rtg = new RandomTreeGenerator();
			rtg.numClassesOption.setValue(numClass);
			rtg.prepareForUse();
			String name = "rtg_" + numClass;
			ThreePair pair = runAndGetDistribution(rtg, N);
			ExpVarPair nxj2 = getExpectationAndVarianceForGammaNXj2(pair.dist);
			ExpVarPair kxj = getExpectationAndVarianceForGammaKXj(pair.dist);
			System.out.println(name + "," + nxj2.expectation + "," + nxj2.variance + "," + kxj.expectation + "," + kxj.variance);
			
			RandomRBFGenerator rbf = new RandomRBFGenerator();
			rbf.numClassesOption.setValue(numClass);
			rbf.numCentroidsOption.setValue(10);
			rbf.prepareForUse();
			name = "rbf_" + numClass + "_10";
			pair = runAndGetDistribution(rbf, N);
			nxj2 = getExpectationAndVarianceForGammaNXj2(pair.dist);
			kxj = getExpectationAndVarianceForGammaKXj(pair.dist);
			System.out.println(name + "," + nxj2.expectation + "," + nxj2.variance + "," + kxj.expectation + "," + kxj.variance);
			
			RandomRBFGenerator rbf2 = new RandomRBFGenerator();
			rbf2.numClassesOption.setValue(numClass);
			rbf2.numCentroidsOption.setValue(50);
			rbf2.prepareForUse();
			name = "rbf_" + numClass + "_50";
			pair = runAndGetDistribution(rbf2, N);
			nxj2 = getExpectationAndVarianceForGammaNXj2(pair.dist);
			kxj = getExpectationAndVarianceForGammaKXj(pair.dist);
			System.out.println(name + "," + nxj2.expectation + "," + nxj2.variance + "," + kxj.expectation + "," + kxj.variance);		
			
			HyperplaneGenerator hpg = new HyperplaneGenerator();
			hpg.numClassesOption.setValue(numClass);
			hpg.prepareForUse();
			name = "hpg_" + numClass;
			pair = runAndGetDistribution(rbf, N);
			nxj2 = getExpectationAndVarianceForGammaNXj2(pair.dist);
			kxj = getExpectationAndVarianceForGammaKXj(pair.dist);
			System.out.println(name + "," + nxj2.expectation + "," + nxj2.variance + "," + kxj.expectation + "," + kxj.variance);
			
			//ArffFileStream covType = new ArffFileStream();
			//covType.arffFileOption.setValue("/Users/cjb60/github/590-cyber-sec/data/covtypeNorm.arff");
			
		
		}
		
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
