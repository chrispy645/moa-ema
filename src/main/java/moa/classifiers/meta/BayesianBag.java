package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import weka.core.Instance;
import weka.core.Utils;
import moa.core.DoubleVector;
import moa.core.Measurement;
import moa.core.MiscUtils;
import moa.options.ClassOption;
import moa.options.FloatOption;
import moa.options.IntOption;

public class BayesianBag extends AbstractClassifier {

    @Override
    public String getPurposeString() {
        return "Incremental on-line bagging of Oza and Russell.";
    }
        
    private static final long serialVersionUID = 1L;

    public ClassOption baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");

    public IntOption ensembleSizeOption = new IntOption("ensembleSize", 's',
            "The number of models in the bag.", 10, 1, Integer.MAX_VALUE);
    
    public FloatOption kOption = new FloatOption("k", 'k', "k param", 1, 1, Double.MAX_VALUE);
    public FloatOption thetaOption = new FloatOption("theta", 't', "theta param", 1, 1, Double.MAX_VALUE); 

    protected Classifier[] ensemble;
    
    private transient GammaDistribution m_gamma = null;

    @Override
    public void resetLearningImpl() {
        this.ensemble = new Classifier[this.ensembleSizeOption.getValue()];
        Classifier baseLearner = (Classifier) getPreparedClassOption(this.baseLearnerOption);
        baseLearner.resetLearning();
        for (int i = 0; i < this.ensemble.length; i++) {
            this.ensemble[i] = baseLearner.copy();
        }
        m_gamma = new GammaDistribution(kOption.getValue(), thetaOption.getValue());
        m_gamma.reseedRandomGenerator( this.classifierRandom.nextLong() );
    }

    @Override
    public void trainOnInstanceImpl(Instance inst) {
    	double[] gammas = new double[this.ensemble.length];
    	for(int i = 0; i < this.ensemble.length; i++) {
    		gammas[i] = m_gamma.sample();
    		//gammas[i] = MiscUtils.poisson(1, this.classifierRandom);
    	}
    	//Utils.normalize(gammas);  	
        for (int i = 0; i < this.ensemble.length; i++) {
            double k = gammas[i];
        	Instance weightedInst = (Instance) inst.copy();
        	weightedInst.setWeight(k);
        	this.ensemble[i].trainOnInstance(weightedInst);
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
