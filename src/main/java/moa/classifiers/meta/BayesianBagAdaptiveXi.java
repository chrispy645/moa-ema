package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;

public class BayesianBagAdaptiveXi extends AbstractBayesianBag {

	private static final long serialVersionUID = -5070753943841429291L;

	@Override
	protected GammaDistribution getDistribution(Instance inst) {
		int classValue = (int) inst.classValue();
		return new GammaDistribution( m_classFreqs[classValue], 1/m_classFreqs[classValue] );
	}

}