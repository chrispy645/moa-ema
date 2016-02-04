package chris;

import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class SortByAttribute {
	
	public static void main(String[] args) throws Exception {
		
		String filename = args[0];
		DataSource ds = new DataSource(filename);
		Instances data = ds.getDataSet();
		
		int attrIndex = Integer.parseInt(args[1]);
		data.sort(attrIndex);
		
		System.out.println(data);
		
		
	}

}
