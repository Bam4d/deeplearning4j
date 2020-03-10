package org.deeplearning4j.rl4j.learning.configuration;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;


@Data
@SuperBuilder
@EqualsAndHashCode(callSuper = true)
public class AsyncQLearningConfiguration extends QLearningConfiguration implements IAsyncLearningConfiguration {

    final int numThreads;
    final int nStep;

    public int getLearnerUpdateFrequency() {
        return getTargetDqnUpdateFreq();
    }
}
