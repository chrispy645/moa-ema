package moa.classifiers.functions;

import java.util.ArrayList;
import java.util.Arrays;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.SparseInstance;
import weka.core.converters.ConverterUtils.DataSource;

public class TestSparseEMA {
	
	private static double[] nonZero(double[] vector) {
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
	
	private static void test(Instances data) {
		System.out.println(data);	
		SparseInstance inst = (SparseInstance) data.get(2);
		System.out.println( inst.numValues() ); // should be 3
		for(int x = 0; x < inst.numValues(); x++) {
			if( inst.index(x) == inst.classIndex() ) {
				continue;
			}
			System.out.print( inst.index(x) + "," + inst.valueSparse(x) + "  " );
		}
		System.out.println("");
	}
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("datasets/simple2_sparse.arff");
		//DataSource ds = new DataSource("/Users/cjb60/github/590-cyber-sec/data/saul-data-ema-arff/computer-scientists-scientist-17.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		
		// test(data);
		
		EMA ema = new EMA();
		for(Instance inst : data) {
			ema.trainOnInstance(inst);
		}
		
		for(int row = 0; row < data.numAttributes()-1; row++) {
			System.out.println( Arrays.toString( nonZero(ema.getWeightMatrix()[row])) );
		}
		System.out.println("");
		
		
		
	}

}
