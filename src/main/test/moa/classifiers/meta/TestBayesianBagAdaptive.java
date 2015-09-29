package moa.classifiers.meta;

import java.util.Arrays;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class TestBayesianBagAdaptive {

	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("/Users/cjb60/github/weka-pyscript/datasets/iris.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1 );
		
		BayesianBagAdaptive bba = new BayesianBagAdaptive();
		bba.prepareForUse();
		bba.resetLearning();
		bba.setDebug(true);
		
		for(int x = 0; x < data.numInstances(); x++) {
			bba.trainOnInstance(data.get(x));
		}
		
		//System.out.println( Arrays.toString( bba.getClassFreqs() ) );
		//System.out.println( bba.getInstCounts() );
		
		/*
		double[] freqs = new double[] { 0.8, 0.1, 0.05, 0.05 };
		double N = 10;
		double c = 0;
		for(double freq : freqs) {
			c += (freq*freq);
		}
		c = c / ( (N+1)*(N+1) );
		System.out.println("c = " + c);
		
		double[] wts = new double[ freqs.length ];
		for(int x = 0; x < wts.length; x++) {
			GammaDistribution g = new GammaDistribution(freqs[x]/c, 1.0/N);
			wts[x] = g.sample();
			System.out.println("params = (" + freqs[x]/c + "," + 1.0/N + "), weight: " + wts[x]);
		}
		Utils.normalize(wts);
		System.out.println( Arrays.toString(wts) );
		*/
		
	}
	
}
