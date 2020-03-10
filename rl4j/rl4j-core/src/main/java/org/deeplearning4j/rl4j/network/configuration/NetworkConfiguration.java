package org.deeplearning4j.rl4j.network.configuration;

import lombok.Builder;
import lombok.Data;
import lombok.NoArgsConstructor;
import lombok.Singular;
import lombok.experimental.SuperBuilder;
import org.deeplearning4j.optimize.api.TrainingListener;
import org.nd4j.linalg.learning.config.IUpdater;

import java.util.List;


@Data
@SuperBuilder
@NoArgsConstructor
public abstract class NetworkConfiguration {

    @Builder.Default
    double learningRate = 0.01;

    @Builder.Default
    double l2 = 0.0;

    IUpdater updater;

    @Singular
    List<TrainingListener> listeners;

}
