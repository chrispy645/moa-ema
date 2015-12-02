package chris;

import java.text.DecimalFormat;
import java.util.Arrays;

import weka.core.Instances;
import weka.core.Utils;
import weka.core.converters.ConverterUtils.DataSource;

public class PrintClassDistribution {
	
	public static void main(String[] args) throws Exception {
		
		DecimalFormat df = new DecimalFormat("#.##");
		
		DataSource ds = new DataSource(args[0]);
		int maxInsts = Integer.parseInt(args[1]);
		Instances data = ds.getDataSet();
		data.setClassIndex(data.numAttributes()-1);
		double[] classDist = new double[ data.numClasses() ];
		for(int x = 0; x < Math.min(maxInsts, data.numInstances()); x++) {
			classDist[ (int) data.get(x).classValue() ] += 1;
			if( x % 1000 == 0) {
				double[] norm = new double[classDist.length];
				for(int i = 0; i < classDist.length; i++) norm[i] = classDist[i];
				Utils.normalize(norm);
				for(double n : norm) {
					System.out.print(df.format(n) + ",");
				}
				System.out.println("");
			}
		}
	}

}
