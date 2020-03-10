package org.deeplearning4j.rl4j.learning.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = false)
public class QLearningConfiguration extends LearningConfiguration {

    @Builder.Default
    int expRepMaxSize = 150000;

    @Builder.Default
    int batchSize = 32;

    @Builder.Default
    int targetDqnUpdateFreq = 100;

    @Builder.Default
    int updateStart = 10;

    @Builder.Default
    double errorClamp = 1.0;

    @Builder.Default
    double minEpsilon = 0.1f;

    @Builder.Default
    int epsilonNbStep = 10000;

    @Builder.Default
    boolean doubleDQN = false;

}
