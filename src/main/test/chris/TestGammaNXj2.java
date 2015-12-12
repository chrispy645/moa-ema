package chris;

import java.util.ArrayList;
import java.util.Arrays;
import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;
import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class TestGammaNXj2 {
	
	public static void main(String[] args) throws Exception {
		
		DataSource ds = new DataSource("/Volumes/CB_RESEARCH/590-cyber-sec-big-output/class-dist-drift-datasets/rbf_5.arff");
		//DataSource ds = new DataSource("/tmp/rbf5.arff");
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		
		ArrayList< ArrayList<Double> > arr = new ArrayList< ArrayList<Double> >();
		for(int k = 0; k < data.numClasses(); k++) arr.add(k, new ArrayList<Double>());
		
		
		//ArrayList<Double> arr = new ArrayList<Double>();
		double[] freqs = new double[data.numClasses()];
		for(int x = 0; x < freqs.length; x++) freqs[x]=1;
		double total = 1;
		for(int x = 0; x < 100000; x++) {
			Instance inst = data.get(x);
			int classVal = (int)inst.classValue();
			
			GammaDistribution g = new GammaDistribution(freqs[classVal], total / (freqs[classVal]*freqs[classVal]) );
			
			freqs[classVal] += 1;
			total += 1;

			arr.get(classVal).add(g.sample());
		}
		
		System.out.println("freqs: " + Arrays.toString(freqs));
		System.out.println(Utils.sum(freqs));
		
		for(int k = 0; k < data.numClasses(); k++) {
		
			double[] tmp = new double[ arr.get(k).size() ];
			for(int x = 0; x < tmp.length; x++) tmp[x] = arr.get(k).get(x);
			
			System.out.println( "empirical mean: " + Utils.mean(tmp) );
			System.out.println( "theoretical mean: " + ( total / freqs[k]) );
		
		}
		
		
	}

}
