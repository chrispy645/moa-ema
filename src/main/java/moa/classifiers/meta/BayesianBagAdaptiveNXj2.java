package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;

public class BayesianBagAdaptiveNXj2 extends AbstractBayesianBag {

	private static final long serialVersionUID = -5073384747355702009L;

	@Override
	protected GammaDistribution getDistribution(Instance inst) {
		int classValue = (int) inst.classValue();
		return new GammaDistribution( m_classFreqs[classValue], 
				m_instCounts / (m_classFreqs[classValue]*m_classFreqs[classValue]) );
	}
	
}