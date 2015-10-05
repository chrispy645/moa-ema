package moa.classifiers.functions;

import org.apache.commons.math3.distribution.GammaDistribution;

public class TestGamma {
	
	public static void main(String[] args) {
		
		GammaDistribution g = new GammaDistribution(1, 1);
		System.out.println(g.sample());
		
	}

}
