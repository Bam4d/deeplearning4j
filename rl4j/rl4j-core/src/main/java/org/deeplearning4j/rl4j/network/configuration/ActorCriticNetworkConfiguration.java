package org.deeplearning4j.rl4j.network.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.EqualsAndHashCode;
import lombok.NoArgsConstructor;
import lombok.experimental.SuperBuilder;

@Data
@SuperBuilder
@NoArgsConstructor
@EqualsAndHashCode(callSuper = true)
public class ActorCriticNetworkConfiguration extends NetworkConfiguration {

    @Builder.Default
    boolean useLSTM = false;

}
