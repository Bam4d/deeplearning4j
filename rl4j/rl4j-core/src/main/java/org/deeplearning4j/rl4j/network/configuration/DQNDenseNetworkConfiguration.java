package org.deeplearning4j.rl4j.network.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@EqualsAndHashCode(callSuper = true)
public class DQNDenseNetworkConfiguration extends NetworkConfiguration {

    @Builder.Default
    int numLayers = 3;

    @Builder.Default
    int numHiddenNodes = 100;
}
