package moa.classifiers.meta;

import org.apache.commons.math3.distribution.GammaDistribution;

import weka.core.Instance;

public class BayesianBagAdaptiveKSqrtK extends AbstractBayesianBag {

	private static final long serialVersionUID = -2450826427881923297L;
	
	private GammaDistribution m_gamma = null;

	@Override
	protected GammaDistribution getDistribution(Instance inst) {
		if(m_gamma == null) {
			double k = inst.numClasses();
			m_gamma = new GammaDistribution( k * Math.sqrt(k), 1.0 / Math.sqrt(k) );
		}
		return m_gamma;
	}

}
