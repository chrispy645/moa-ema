package chris;


import java.io.BufferedWriter;
import java.io.File;
import java.io.FileWriter;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Discretize;
import weka.filters.unsupervised.attribute.NominalToBinary;
import weka.filters.unsupervised.attribute.NumericToBinary;
import weka.filters.unsupervised.attribute.RemoveUseless;
import weka.filters.unsupervised.instance.NonSparseToSparse;
import moa.streams.generators.RandomRBFGeneratorDrift;

public class GenerateRBFDataset {
	
	private static void binarise(String filename, String outFile) throws Exception {
		DataSource ds = new DataSource(filename);
		Instances data = ds.getDataSet();
		data.setClassIndex( data.numAttributes() - 1);
		
		/*
		RemoveUseless ru = new RemoveUseless();
		ru.setInputFormat(data);
		data = Filter.useFilter(data, ru);
		*/
		
		Discretize dis = new Discretize();
		dis.setInputFormat(data);
		dis.setBins(10);
		Instances newData = Filter.useFilter(data, dis);
		
		NominalToBinary nom = new NominalToBinary();
		nom.setInputFormat(newData);
		newData = Filter.useFilter(newData, nom);
		
		NumericToBinary nn = new NumericToBinary();
		nn.setInputFormat(newData);
		newData = Filter.useFilter(newData, nn);
		
		NonSparseToSparse sp = new NonSparseToSparse();
		sp.setInputFormat(newData);
		newData = Filter.useFilter(newData, sp);
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File(outFile));
		saver.writeBatch();
		
	}
	
	public static void main(String[] args) throws Exception {
		
		/*
		 * Some classes have 0 occurences. Make each weight
		 * the same?
		 */
		
		/*
		rbf.numDriftCentroidsOption.setValue(10);
		rbf.speedChangeOption.setValue(1);
		*/
		
		//int[] numClasses = new int[] { 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096 };
		//int[] numAttrs = new int[] { 2, 4, 8, 16, 32, 64, 128, 256 };
		
		int[] numClasses = new int[] { 128 };
		int[] numAttrs = new int[] { 64 };
		
		for(int cls : numClasses) {
			for(int attr : numAttrs) {
				//if( new File("datasets/rbf/" + cls + "_" + attr + ".arff").exists())
				//	continue;			
				
				System.out.println("Processing: " + cls + "," + attr);
				RandomRBFGeneratorDrift rbf = new RandomRBFGeneratorDrift();
				rbf.instanceRandomSeedOption.setValue(0);
				rbf.modelRandomSeedOption.setValue(0);
				
				rbf.numDriftCentroidsOption.setValue(cls);
				
				BufferedWriter writer = new BufferedWriter(new FileWriter("/tmp/" + cls + "_" + attr + ".arff") );
				rbf.numClassesOption.setValue(cls);
				rbf.numCentroidsOption.setValue(cls);
				rbf.numAttsOption.setValue(attr);
				rbf.prepareForUse();
				writer.write( rbf.getHeader() + "\n" );
				for(int x = 0; x < 100000; x++) {
					writer.write( rbf.nextInstance() + "\n" );
				}
				writer.flush();
				writer.close();			
				binarise("/tmp/" + cls + "_" + attr + ".arff", "datasets/rbf/" + cls + "_" + attr + ".arff");
			}
		}
		
	}

}
