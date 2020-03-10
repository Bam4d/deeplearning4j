package org.deeplearning4j.rl4j.learning.configuration;

import lombok.AllArgsConstructor;
import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@EqualsAndHashCode(callSuper = true)
public class A3CLearningConfiguration extends LearningConfiguration implements IAsyncLearningConfiguration {

    final int numThreads;
    final int nStep;
    int learnerUpdateFrequency = -1;
}
