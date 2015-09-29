package moa.classifiers.meta;

import java.util.Arrays;

import org.apache.commons.math3.distribution.GammaDistribution;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Utils;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.options.IntOption;

public class BayesianBagAdaptive extends AbstractClassifier {

    @Override
    public String getPurposeString() {
        return "Adaptive Bayesian bagging";
    }
        
    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);
    
    protected Classifier[] ensemble;
    
    protected transient GammaDistribution m_gammaDefault = null;
    
    private double[] m_classFreqs = null;
    private int m_instCounts = 0;
    
    private boolean m_debug = true;
    
    public void setDebug(boolean b) {
    	m_debug = b;
    }
    
    public double[] getClassFreqs() {
    	return m_classFreqs;
    }
    
    public int getInstCounts() {
    	return m_instCounts;
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        m_classFreqs = null;
        m_instCounts = 0;
        m_gammaDefault = new GammaDistribution(1,1);
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if(m_classFreqs == null) {
    		m_classFreqs = new double[ inst.numClasses() ];
    		// laplace
    		for(int x = 0; x < m_classFreqs.length; x++) {
    			m_classFreqs[x] = 1;
    		}
    	}
    	int classValue = (int)inst.classValue();
    	m_instCounts += 1;
    	m_classFreqs[classValue] += 1;
    	double[] weights = null;
    	if(m_instCounts < 100) {
	    	weights = new double[this.ensemble.length];
	    	for(int i = 0; i < this.ensemble.length; i++) {
	    		//weights[i] = MiscUtils.poisson(1, this.classifierRandom);
	    		weights[i] = m_gammaDefault.sample();
	    	} 	
    	} else {
    		weights = new double[this.ensemble.length];
    		double[] norm = Arrays.copyOf(m_classFreqs, m_classFreqs.length);
    		Utils.normalize(norm);
    		if(m_debug) {
    			System.err.println( Arrays.toString(norm) );
    		}
    		double c = 0;
    		for(double n : norm) {
    			c += (n*n);
    		}
    		c = c / ((m_instCounts+1)*(m_instCounts+1));
    		
    		// for each model in the ensemble...
    		for (int i = 0; i < this.ensemble.length; i++) {
    			// generate weights from gamma(...) for all classes,
    			// normalise, and then assign model[i] = weight[classValue]
	    		double[] weightsForAllClasses = new double[inst.numClasses()];
	    		for(int x = 0; x < inst.numClasses(); x++) {
	    			if(m_debug) {
	    				System.out.println("norm = " + Arrays.toString(norm) + ", c = " + c + ", params = (" + (norm[x]/c) + ","
	    						+ (1.0/(m_instCounts+1)) + "), weight: " + weightsForAllClasses[x]);
	    			}
	    			GammaDistribution g = new GammaDistribution(norm[x]/c, 1.0/(m_instCounts+1));
	    			weightsForAllClasses[x] = g.sample();
	    		}
	    		Utils.normalize(weightsForAllClasses);
	    		
	        	Instance weightedInst = (Instance) inst.copy();
	        	weightedInst.setWeight( weightsForAllClasses[classValue] );
	        	this.ensemble[i].trainOnInstance(weightedInst);
    		}
    		
    	}	
    }

    @Override
    public double[] getVotesForInstance(Instance inst) {
        DoubleVector combinedVote = new DoubleVector();
        for (int i = 0; i < this.ensemble.length; i++) {
            DoubleVector vote = new DoubleVector(this.ensemble[i].getVotesForInstance(inst));
            if (vote.sumOfValues() > 0.0) {
                vote.normalize();
                combinedVote.addValues(vote);
            }
        }
        return combinedVote.getArrayRef();
    }

    @Override
    public boolean isRandomizable() {
        return true;
    }

    @Override
    public void getModelDescription(StringBuilder out, int indent) {
        // TODO Auto-generated method stub
    }

    @Override
    protected Measurement[] getModelMeasurementsImpl() {
        return new Measurement[]{new Measurement("ensemble size",
                    this.ensemble != null ? this.ensemble.length : 0)};
    }

    @Override
    public Classifier[] getSubClassifiers() {
        return this.ensemble.clone();
    }
}