package org.deeplearning4j.rl4j.support;

import lombok.AllArgsConstructor;
import lombok.Value;
import org.deeplearning4j.rl4j.learning.configuration.IAsyncLearningConfiguration;

@Value
@AllArgsConstructor
public class MockAsyncConfiguration implements IAsyncLearningConfiguration {

    private Long seed;
    private int maxEpochStep;
    private int maxStep;
    private int updateStart;
    private double rewardFactor;
    private double gamma;
    private double errorClamp;
    private int numThreads;
    private int nStep;
    private int learnerUpdateFrequency;
}
