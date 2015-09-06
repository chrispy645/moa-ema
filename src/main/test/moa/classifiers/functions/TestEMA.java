package moa.classifiers.functions;

import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class TestEMA {
	
	/*
	 * [0.15]
	 * [0.258375, 0.1275]
	 * [0.15]
	 * [0.15]
	 */
	
	public static double[] nonZero(double[] vector) {
		ArrayList<Double> tmp = new ArrayList<Double>();
		for(double v : vector) {
			if( v != 0.0) {
				tmp.add(v);
			}
		}
		double[] finalArr = new double[ tmp.size() ];
		for(int x = 0; x < finalArr.length; x++) {
			finalArr[x] = tmp.get(x);
		}
		return finalArr;
	}
	
	public static void printArray(double[][] arr) {
		for(int x = 0; x < arr.length; x++) { 
			System.out.println(Arrays.toString(arr[x]));
		}
	}
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("datasets/simple2.arff");
		//DataSource ds = new DataSource("/Users/cjb60/github/590-cyber-sec/data/saul-data-ema-arff/computer-scientists-scientist-17.arff");
		
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		
		EMA ema = new EMA();
		for(Instance inst : data) {
			ema.trainOnInstance(inst);
		}
		
		printArray( ema.getWeightMatrix() );
		
	}

}
