package moa.classifiers.meta;

import java.util.Arrays;

import moa.streams.generators.RandomRBFGenerator;

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

		/*
		for(int x = 0; x < data.numInstances(); x++) {
			bba.trainOnInstance(data.get(x));
		}
		*/

		RandomRBFGenerator gen = new RandomRBFGenerator();
		gen.prepareForUse();
		
		long t0 = System.currentTimeMillis();
		for(int x = 0; x < 1000; x++) {
			bba.trainOnInstance( gen.nextInstance() );
		}
		long now = System.currentTimeMillis();
		System.out.println("Time: " + (double)(now - t0) / 1000.0 + " secs" );
		
	}
	
}
