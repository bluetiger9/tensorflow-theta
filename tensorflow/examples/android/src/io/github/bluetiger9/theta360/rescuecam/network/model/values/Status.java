package io.github.bluetiger9.theta360.rescuecam.network.model.values;

public enum Status {
    SHOOTING("shooting"),
    IDLE("idle"),;

    private final String mStatus;

    Status(String status) {
        this.mStatus = status;
    }

    @Override
    public String toString() {
        return this.mStatus;
    }
}
