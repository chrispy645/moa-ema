package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.FlagOption;
import moa.options.IntOption;

public abstract class AbstractBayesianBag extends AbstractClassifier {
	
    @Override
    public String getPurposeString() {
        return "Adaptive Bayesian bagging";
    }
        
    private static final long serialVersionUID = 1L;
    
    public FlagOption debugOption = new FlagOption("debug", 'd', "debug");

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);
    
    protected Classifier[] ensemble;
    
    protected transient GammaDistribution m_gammaDefault = null;
    
    protected double[] m_classFreqs = null;
    protected double m_instCounts = 0;
    
    protected boolean m_debug = false;
    
    public void setDebug(boolean b) {
    	m_debug = b;
    }
    
    public double[] getClassFreqs() {
    	return m_classFreqs;
    }
    
    public double getInstCounts() {
    	return m_instCounts;
    }

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        this.m_debug = this.debugOption.isSet();
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        m_classFreqs = null;
        m_instCounts = 1; // laplace
        m_gammaDefault = new GammaDistribution(1,1);
    }
    
    protected abstract GammaDistribution getDistribution(Instance inst);

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	if(m_classFreqs == null) {
    		m_classFreqs = new double[ inst.numClasses() ];
    		for(int x = 0; x < m_classFreqs.length; x++) {
    			m_classFreqs[x] = 1; // laplace
    		}
    	}
    	
    	int classValue = (int)inst.classValue();
		// for each model in the ensemble...
		for (int i = 0; i < this.ensemble.length; i++) {
			GammaDistribution g = getDistribution(inst);	  			
			g.reseedRandomGenerator( classifierRandom.nextLong() );
			double weight = g.sample(); 			   			
			if(m_debug) { 
				System.out.println(weight);
			}   				    		
        	Instance weightedInst = (Instance) inst.copy();
        	//weightedInst.setWeight( weightsForAllClasses[classValue] );
        	weightedInst.setWeight(weight);
        	this.ensemble[i].trainOnInstance(weightedInst);
		}

    	m_instCounts += 1;
    	m_classFreqs[classValue] += 1;
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
