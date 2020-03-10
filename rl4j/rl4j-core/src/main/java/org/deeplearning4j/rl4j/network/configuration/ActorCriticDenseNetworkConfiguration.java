package org.deeplearning4j.rl4j.network.configuration;

import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@EqualsAndHashCode(callSuper = true)
public class ActorCriticDenseNetworkConfiguration extends ActorCriticNetworkConfiguration {
    int numLayers;
    int numHiddenNodes;
}
