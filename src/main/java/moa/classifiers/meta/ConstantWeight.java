package moa.classifiers.meta;

import weka.core.Instance;
import moa.classifiers.AbstractClassifier;
import moa.classifiers.Classifier;
import moa.core.Measurement;
import moa.options.ClassOption;
import moa.options.IntOption;

public class ConstantWeight extends AbstractClassifier {	
	
	private static final long serialVersionUID = 3300361702916945950L;
	
	public ClassOption m_baseLearnerOption = new ClassOption("baseLearner", 'l',
            "Classifier to train.", Classifier.class, "trees.HoeffdingTree");
	private Classifier m_baseLearner = null;
	public IntOption m_weightOption = new IntOption("weight", 'w', "weight", 1, 1, 20);
	private int m_weight = 1;

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		return m_baseLearner.getVotesForInstance(inst);
	}

	@Override
	public void resetLearningImpl() {
		m_baseLearner = (Classifier) getPreparedClassOption(m_baseLearnerOption);
		m_weight = m_weightOption.getValue();
	}

	@Override
	public void trainOnInstanceImpl(Instance inst) {
		Instance copyInst = (Instance) inst.copy();
		copyInst.setWeight(m_weight);
		m_baseLearner.trainOnInstance(copyInst);
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
	}

}
