package moa.classifiers.functions;

import moa.classifiers.AbstractClassifier;
import moa.core.Measurement;
import weka.core.Instance;
import weka.core.Utils;

/**
 * Exponential moving average algorithm. This implementation
 * is for dense instances, so it might be quite slow.
 * @author cjb60
 *
 */
public class EMA extends AbstractClassifier {

	private static final long serialVersionUID = -4357871674824496401L;

	private double[][] m_weightMatrix = null;
	
	private double beta = 0.15;
	private double dm = 0.15;
	private double wmin = 0.01;

	@Override
	public boolean isRandomizable() {
		return false;
	}

	@Override
	public double[] getVotesForInstance(Instance inst) {
		if( m_weightMatrix == null ) {
			m_weightMatrix = new double[inst.numAttributes()-1][inst.numClasses()];
		}
		// score the connected classes
		double[] scores = new double[inst.numClasses()];
		for(int c = 0; c < inst.numClasses(); c++) {
			// iterate through the instance x
			double sum = 0;
			for(int f = 0; f < inst.numAttributes()-1; f++) {
				if( inst.value(f) == 0.0) {
					continue;
				}
				sum += (inst.value(f) * m_weightMatrix[f][c]);
			}
			scores[c] = sum;
		}
		try {
			Utils.normalize(scores);
		} catch(Exception ex) {
			return new double[ inst.numClasses() ];
		}
		return scores;
	}

	@Override
	public void resetLearningImpl() {
		m_weightMatrix = null;		
	}
	
	@Override
	public void trainOnInstanceImpl(Instance inst) {
		if( m_weightMatrix == null ) {
			m_weightMatrix = new double[inst.numAttributes()-1][inst.numClasses()];
		}
		// score the connected classes
		double[] scores = new double[inst.numClasses()];
		for(int c = 0; c < inst.numClasses(); c++) {
			// iterate through the instance x
			double sum = 0;
			for(int f = 0; f < inst.numAttributes()-1; f++) {
				if( inst.value(f) == 0.0) {
					continue;
				}
				sum += (inst.value(f) * m_weightMatrix[f][c]);
			}
			scores[c] = sum;
		}
		
		// ok, get the score of the actual class, and the score
		// of the most common class that's not the actual class
		double scoreOfActualClass = scores[ (int)inst.classValue() ];
		double scorePrime = scores[0];
		int[] sortedIndices = Utils.sort(scores);
		for(int i = sortedIndices.length-1; i >= 0; i--) {
			if( sortedIndices[i] != (int)inst.classValue() ) {
				scorePrime = scores[ sortedIndices[i] ];
				break;
			}
		}
		
		// compute the margin
		// if margin is not met, then update
		double dx = scoreOfActualClass - scorePrime;	
		if( dx < dm ) {
			// for all f in x, do...
			for(int f = 0; f < inst.numAttributes()-1; f++) {
				if( inst.value(f) == 0.0 ) {
					continue;
				}
				// all active features' connections are first decayed,
				// then the connections to the true class are boosted
				// 3.1
				for(int c = 0; c < inst.numClasses(); c++) {
					//if( c == x.classValue() ) continue;
					// (1 - x_f^2 * beta) * w_fc
					m_weightMatrix[f][c] = ( 1 - (inst.value(f)*inst.value(f)*beta) ) * m_weightMatrix[f][c];
				}
				// 3.2
				// boost connection to the true class
				m_weightMatrix[f][(int)inst.classValue()] += (inst.value(f)*beta);
				// 3.3
				// then, drop any tiny weights
				for(int c = 0; c < inst.numClasses(); c++) {
					if( m_weightMatrix[f][c] < wmin) {
						m_weightMatrix[f][c] = 0;
					}
				}
			}
		}	
		
	}

	@Override
	protected Measurement[] getModelMeasurementsImpl() {
		// TODO Auto-generated method stub
		return null;
	}

	@Override
	public void getModelDescription(StringBuilder out, int indent) {
		// TODO Auto-generated method stub
		
	}

}
