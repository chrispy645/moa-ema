package moa.streams.generators;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Random;

public class RandomRBFGeneratorNR extends RandomRBFGenerator {
	
	private static final long serialVersionUID = -9182305767310679423L;

	@Override
    protected void generateCentroids() {
        Random modelRand = new Random(this.modelRandomSeedOption.getValue());
        this.centroids = new Centroid[this.numCentroidsOption.getValue()];
        this.centroidWeights = new double[this.centroids.length];
        
        ArrayList<Integer> classIndices = new ArrayList<Integer>();
        for(int x = 0; x < Math.ceil(((double) this.numCentroidsOption.getValue()) / ((double) this.numClassesOption.getValue()) ); x++) {
        	for(int y = 0; y < this.numClassesOption.getValue(); y++) {
        		classIndices.add(y);
        	}
        }
        Collections.shuffle(classIndices, modelRand);
        
        for (int i = 0; i < this.centroids.length; i++) {
            this.centroids[i] = new Centroid();
            double[] randCentre = new double[this.numAttsOption.getValue()];
            for (int j = 0; j < randCentre.length; j++) {
                randCentre[j] = modelRand.nextDouble();
            }
            this.centroids[i].centre = randCentre;
            //this.centroids[i].classLabel = modelRand.nextInt(this.numClassesOption.getValue());
            this.centroids[i].classLabel = classIndices.get(i);
            this.centroids[i].stdDev = modelRand.nextDouble();
            this.centroidWeights[i] = modelRand.nextDouble();
        }
    }

}
