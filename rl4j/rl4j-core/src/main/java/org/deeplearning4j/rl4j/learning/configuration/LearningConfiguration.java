package org.deeplearning4j.rl4j.learning.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
public class LearningConfiguration implements ILearningConfiguration {

    @Builder.Default
    Long seed = System.currentTimeMillis();

    @Builder.Default
    int maxEpochStep = 200;

    @Builder.Default
    int maxStep = 150000;

    @Builder.Default
    double gamma = 0.99;

    @Builder.Default
    double rewardFactor = 1.0;

}
