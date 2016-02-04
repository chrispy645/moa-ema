package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;

public class BayesianBagAdaptiveKXj extends AbstractBayesianBag {

	private static final long serialVersionUID = -3587754156697423749L;

	@Override
	protected GammaDistribution getDistribution(Instance inst) {
		int classValue = (int) inst.classValue();
		double k = inst.numClasses();
		return new GammaDistribution( m_classFreqs[classValue], k / m_classFreqs[classValue] );	
	}

}