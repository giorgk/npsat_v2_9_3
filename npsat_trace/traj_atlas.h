#ifndef NPSAT_V2_TRACE_TRAJ_ATLAS_H
#define NPSAT_V2_TRACE_TRAJ_ATLAS_H

#include <deal.II/base/exceptions.h>
#include <deal.II/base/point.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <map>
#include <vector>

namespace npsat_trace {
    using namespace dealii;
    /**
    * Cell-local trajectory atlas data structures.
    *
    * Sign convention: subface_flow and well_flow are forward flows, positive
    * outward from the aquifer cell.  An atlas trajectory starts just inside a
    * forward-outflow subface, or at a forward aquifer-outflow well, and is
    * integrated backward until it reaches a forward-inflow subface, a well, or
    * an unresolved terminal.
    *
    * The atlas is intentionally independent of the velocity reconstruction and
    * of MPI.  It is built and queried only on the rank owning the cell.  Packet
    * migration remains the responsibility of the tracing driver.
    */

    static const unsigned int atlas_faces_per_cell = 6;
    static const unsigned int atlas_slots_per_face = 4;
    static const unsigned int atlas_max_subfaces = atlas_faces_per_cell * atlas_slots_per_face;

    inline unsigned int atlas_subface_id(const unsigned int face,
                                        const unsigned int slot)
    {
        AssertIndexRange(face, atlas_faces_per_cell);
        AssertIndexRange(slot, atlas_slots_per_face);
        return face * atlas_slots_per_face + slot;
    }

    inline unsigned int atlas_face_from_subface_id(const unsigned int id)
    {
        AssertIndexRange(id, atlas_max_subfaces);
        return id / atlas_slots_per_face;
    }

    inline unsigned int atlas_slot_from_subface_id(const unsigned int id)
    {
        AssertIndexRange(id, atlas_max_subfaces);
        return id % atlas_slots_per_face;
    }

    enum class AtlasTerminalKind : std::uint8_t
    {
        subface = 0,
        well = 1,
        stagnant = 2,
        unresolved = 3
    };

enum class AtlasBuildState : std::uint8_t
    {
        empty = 0,
        collecting = 1,
        finalized = 2
};

enum class AtlasPacketState : std::uint8_t
{
    active = 0,
    step_complete = 1,
    terminated = 2,
    unresolved = 3
};

    enum class AtlasOriginKind : std::uint8_t
    {
        subface = 0,
        well = 1
    };

    /** Downstream boundary from which a backward atlas trajectory starts. */
    struct AtlasOrigin
    {
        AtlasOriginKind kind = AtlasOriginKind::subface;
        std::uint32_t id = 0;

        static AtlasOrigin subface(const unsigned int subface_id)
        {
            AssertIndexRange(subface_id, atlas_max_subfaces);
            AtlasOrigin value;
            value.kind = AtlasOriginKind::subface;
            value.id = static_cast<std::uint32_t>(subface_id);
            return value;
        }

        static AtlasOrigin well(const std::uint32_t well_id)
        {
            AtlasOrigin value;
            value.kind = AtlasOriginKind::well;
            value.id = well_id;
            return value;
        }

        bool operator<(const AtlasOrigin &other) const
        {
            if (kind != other.kind)
                return static_cast<unsigned int>(kind) <
                    static_cast<unsigned int>(other.kind);
            return id < other.id;
        }
    };

    /** Identifies an upstream terminal reached by a backward trajectory. */
    struct AtlasTerminal
    {
        AtlasTerminalKind kind = AtlasTerminalKind::unresolved;
        std::uint32_t id = 0;

        static AtlasTerminal subface(const unsigned int subface_id)
        {
            AssertIndexRange(subface_id, atlas_max_subfaces);
            AtlasTerminal value;
            value.kind = AtlasTerminalKind::subface;
            value.id = static_cast<std::uint32_t>(subface_id);
            return value;
        }

        static AtlasTerminal well(const std::uint32_t well_id)
        {
            AtlasTerminal value;
            value.kind = AtlasTerminalKind::well;
            value.id = well_id;
            return value;
        }

        static AtlasTerminal stagnant()
        {
            AtlasTerminal value;
            value.kind = AtlasTerminalKind::stagnant;
            return value;
        }

        static AtlasTerminal unresolved()
        {
            return AtlasTerminal();
        }

        bool operator==(const AtlasTerminal &other) const
        {
            return kind == other.kind && id == other.id;
        }

        bool operator<(const AtlasTerminal &other) const
        {
            if (kind != other.kind)
                return static_cast<unsigned int>(kind) <
                    static_cast<unsigned int>(other.kind);
            return id < other.id;
        }
    };

    /** Geometry and forward-flow data for one canonical outer subface. */
    template <int dim>
    struct AtlasSubface
    {
        static_assert(dim == 3, "The trajectory atlas currently assumes dim == 3.");

        bool active = false;
        unsigned char face = 0;
        unsigned char slot = 0;
        Point<dim> center;
        double area = 0.0;
        double outward_flow = 0.0;

        unsigned int id() const
        {
            return atlas_subface_id(face, slot);
        }

        bool is_forward_inflow(const double tolerance = 0.0) const
        {
            return active && outward_flow < -tolerance;
        }

        bool is_forward_outflow(const double tolerance = 0.0) const
        {
            return active && outward_flow > tolerance;
        }
    };

    /** Forward aquifer-to-well exchange for a well intersecting this cell. */
struct AtlasWellTerminal
    {
        std::uint32_t well_id = 0;
        double outward_flow = 0.0;
};

/**
 * Persistent state of one flow branch.  Length is aquifer arc length only;
 * instantaneous well-bore routing must not add to it.
 */
template <int dim>
struct AtlasPacket
{
    std::uint64_t receptor_id = 0;
    std::uint64_t seed_id = 0;
    std::uint64_t branch_id = 0;
    std::uint64_t parent_branch_id = 0;
    dealii::Point<dim> position;
    double weight = 1.0;
    double age = 0.0;
    double aquifer_length = 0.0;
    double dt_remaining = 0.0;
    unsigned int flow_step = 0;
    unsigned int generation = 0;
    AtlasPacketState state = AtlasPacketState::active;
};

/** Fixed-layout representation exchanged with MPI_BYTE. */
struct AtlasPacketWire
{
    std::uint64_t receptor_id;
    std::uint64_t seed_id;
    std::uint64_t branch_id;
    std::uint64_t parent_branch_id;
    double x;
    double y;
    double z;
    double weight;
    double age;
    double aquifer_length;
    double dt_remaining;
    std::uint32_t flow_step;
    std::uint32_t generation;
    std::uint8_t state;
};

template <int dim>
inline AtlasPacketWire pack_atlas_packet(const AtlasPacket<dim> &packet)
{
    static_assert(dim == 3, "Atlas packet wire format is three-dimensional.");
    AtlasPacketWire wire;
    wire.receptor_id = packet.receptor_id;
    wire.seed_id = packet.seed_id;
    wire.branch_id = packet.branch_id;
    wire.parent_branch_id = packet.parent_branch_id;
    wire.x = packet.position[0];
    wire.y = packet.position[1];
    wire.z = packet.position[2];
    wire.weight = packet.weight;
    wire.age = packet.age;
    wire.aquifer_length = packet.aquifer_length;
    wire.dt_remaining = packet.dt_remaining;
    wire.flow_step = static_cast<std::uint32_t>(packet.flow_step);
    wire.generation = static_cast<std::uint32_t>(packet.generation);
    wire.state = static_cast<std::uint8_t>(packet.state);
    return wire;
}

template <int dim>
inline AtlasPacket<dim> unpack_atlas_packet(const AtlasPacketWire &wire)
{
    static_assert(dim == 3, "Atlas packet wire format is three-dimensional.");
    AtlasPacket<dim> packet;
    packet.receptor_id = wire.receptor_id;
    packet.seed_id = wire.seed_id;
    packet.branch_id = wire.branch_id;
    packet.parent_branch_id = wire.parent_branch_id;
    packet.position = dealii::Point<dim>(wire.x, wire.y, wire.z);
    packet.weight = wire.weight;
    packet.age = wire.age;
    packet.aquifer_length = wire.aquifer_length;
    packet.dt_remaining = wire.dt_remaining;
    packet.flow_step = wire.flow_step;
    packet.generation = wire.generation;
    packet.state = static_cast<AtlasPacketState>(wire.state);
    return packet;
}

template <int dim>
struct AtlasQuadraturePoint
{
    dealii::Point<dim> position;
    double weight = 0.0;
};

template <int dim>
struct AtlasOuterSubfaceData
{
    unsigned char face = 0;
    unsigned char slot = 0;
    dealii::Point<dim> center;
    std::array<dealii::Point<dim>, 4> vertices;
    double area = 0.0;
    double outward_normal_velocity = 0.0;

    double outward_flow() const
    {
        return area * outward_normal_velocity;
    }
};

    /**
    * One time-parameterized sample along a backward local trajectory.
    * backward_time and aquifer_length are measured from the trajectory's
    * forward-outflow starting point.  Both must be nondecreasing.
    */
    template <int dim>
    struct AtlasTrajectorySample
    {
        Point<dim> position;
        double backward_time = 0.0;
        double aquifer_length = 0.0;
    };

    /** A flow-weighted local trajectory used by the atlas. */
    template <int dim>
struct AtlasTrajectory
    {
        static_assert(dim == 3, "The trajectory atlas currently assumes dim == 3.");

        std::uint32_t id = 0;
        AtlasOrigin downstream_origin;
        AtlasTerminal upstream_terminal;

        // Portion of the downstream subface or well flow represented here.
        double represented_flow = 0.0;
        std::vector<AtlasTrajectorySample<dim> > samples;

        double total_time() const
        {
            return samples.empty() ? 0.0 : samples.back().backward_time;
        }

        double total_length() const
        {
            return samples.empty() ? 0.0 : samples.back().aquifer_length;
        }

        bool valid(const double tolerance = 1.0e-12) const
        {
            if ((downstream_origin.kind == AtlasOriginKind::subface &&
                downstream_origin.id >= atlas_max_subfaces) ||
                !std::isfinite(represented_flow) || represented_flow < 0.0 ||
                samples.empty())
                return false;

            for (unsigned int i = 0; i < samples.size(); ++i)
            {
                const AtlasTrajectorySample<dim> &sample = samples[i];
                if (!std::isfinite(sample.backward_time) ||
                    !std::isfinite(sample.aquifer_length) ||
                    sample.backward_time < -tolerance ||
                    sample.aquifer_length < -tolerance)
                    return false;

                for (unsigned int d = 0; d < dim; ++d)
                    if (!std::isfinite(sample.position[d]))
                        return false;

                if (i > 0 &&
                    (sample.backward_time + tolerance < samples[i - 1].backward_time ||
                    sample.aquifer_length + tolerance < samples[i - 1].aquifer_length))
                    return false;
            }
            return true;
        }

        /** Interpolate position and accumulated length at a stored backward time. */
        AtlasTrajectorySample<dim> sample_at_time(const double time) const
        {
            AssertThrow(!samples.empty(),
                        dealii::ExcMessage("Cannot sample an empty atlas trajectory."));

            if (time <= samples.front().backward_time)
                return samples.front();
            if (time >= samples.back().backward_time)
                return samples.back();

            typename std::vector<AtlasTrajectorySample<dim> >::const_iterator upper =
                std::lower_bound(
                    samples.begin(), samples.end(), time,
                    [](const AtlasTrajectorySample<dim> &sample, const double value) {
                        return sample.backward_time < value;
                    });

            const AtlasTrajectorySample<dim> &b = *upper;
            const AtlasTrajectorySample<dim> &a = *(upper - 1);
            const double dt = b.backward_time - a.backward_time;
            const double alpha = dt > 0.0 ? (time - a.backward_time) / dt : 0.0;

            AtlasTrajectorySample<dim> result;
            result.backward_time = time;
            result.aquifer_length =
                a.aquifer_length + alpha * (b.aquifer_length - a.aquifer_length);
            for (unsigned int d = 0; d < dim; ++d)
                result.position[d] =
                    a.position[d] + alpha * (b.position[d] - a.position[d]);
            return result;
        }
};

inline std::uint64_t atlas_child_branch_id(const std::uint64_t parent,
                                           const unsigned int step,
                                           const unsigned int child)
{
    std::uint64_t value = parent;
    value ^= static_cast<std::uint64_t>(step) + 0x9e3779b97f4a7c15ULL +
             (value << 6) + (value >> 2);
    value ^= static_cast<std::uint64_t>(child) + 0x9e3779b97f4a7c15ULL +
             (value << 6) + (value >> 2);
    return value;
}

/** Resample a numerically integrated path at uniformly spaced times. */
template <int dim>
inline std::vector<AtlasTrajectorySample<dim> > resample_atlas_trajectory(
    const std::vector<AtlasTrajectorySample<dim> > &input,
    const unsigned int requested)
{
    std::vector<AtlasTrajectorySample<dim> > output;
    if (input.empty())
        return output;
    if (input.size() <= requested || requested < 2)
        return input;

    output.reserve(requested);
    const double total_time = input.back().backward_time;
    for (unsigned int i = 0; i < requested; ++i)
    {
        const double target = total_time * static_cast<double>(i) /
                              static_cast<double>(requested - 1);
        typename std::vector<AtlasTrajectorySample<dim> >::const_iterator upper =
            std::lower_bound(
                input.begin(), input.end(), target,
                [](const AtlasTrajectorySample<dim> &sample, const double time) {
                    return sample.backward_time < time;
                });
        if (upper == input.begin())
        {
            output.push_back(*upper);
            continue;
        }
        if (upper == input.end())
        {
            output.push_back(input.back());
            continue;
        }
        const AtlasTrajectorySample<dim> &b = *upper;
        const AtlasTrajectorySample<dim> &a = *(upper - 1);
        const double dt = b.backward_time - a.backward_time;
        const double alpha = dt > 0.0 ? (target - a.backward_time) / dt : 0.0;
        AtlasTrajectorySample<dim> sample;
        sample.backward_time = target;
        sample.aquifer_length = a.aquifer_length +
            alpha * (b.aquifer_length - a.aquifer_length);
        for (unsigned int d = 0; d < dim; ++d)
            sample.position[d] = a.position[d] +
                alpha * (b.position[d] - a.position[d]);
        output.push_back(sample);
    }
    return output;
}

/** User-controlled atlas construction and query tolerances. */
    struct TrajectoryAtlasOptions
    {
        unsigned int quadrature_points_per_subface = 4;
        unsigned int stored_samples_per_trajectory = 8;
        unsigned int maximum_query_samples = 32;

        double kernel_power = 2.0;
        double kernel_epsilon = 1.0e-6;
        double flow_tolerance = 1.0e-12;
        double balance_relative_tolerance = 1.0e-6;
        double minimum_branch_fraction = 1.0e-6;

        // Physical coordinate scales used by the anisotropic query metric.
        std::array<double, 3> metric_scale{{1.0, 1.0, 1.0}};
    };

    /** A flow entry in the local downstream-to-upstream transfer table. */
    struct AtlasTransferEntry
    {
        AtlasOrigin downstream_origin;
        AtlasTerminal upstream_terminal;
        double flow = 0.0;
        double fraction = 0.0;
    };

    /** Result for one terminal branch returned by an interior atlas query. */
    template <int dim>
    struct AtlasQueryBranch
    {
        AtlasTerminal terminal;
        double fraction = 0.0;

        // Conditional values from the query point to the terminal.
        double remaining_time = 0.0;
        double remaining_length = 0.0;

        // State after min(available_time, remaining_time).  If reaches_terminal is
        // true this is the terminal point; otherwise it is an interior point for
        // the next transient velocity step.
        bool reaches_terminal = false;
        dealii::Point<dim> advanced_position;
        double advanced_time = 0.0;
        double advanced_length = 0.0;
    };

    template <int dim>
    struct AtlasQueryResult
    {
        bool valid = false;
        std::vector<AtlasQueryBranch<dim> > branches;
        double retained_fraction = 0.0;
        double discarded_fraction = 0.0;
    };

/**
 * Read-mostly trajectory atlas for one locally owned cell and flow step.
 *
 * Building trajectories is deliberately outside this class: the caller owns
 * the velocity evaluator, event integration, and well capture model.  This
 * class owns their reusable products, validates them, builds transfer totals,
 * and performs flow-weighted interior queries.
 */
template <int dim>
class CellTrajectoryAtlas
    {
    public:
        static_assert(dim == 3, "The trajectory atlas currently assumes dim == 3.");

        CellTrajectoryAtlas()
        {
            clear();
        }

        void clear()
        {
            state_ = AtlasBuildState::empty;
            cell_id_ = 0;
            flow_step_ = 0;
            tracking_direction_ = -1;
            subfaces_ = std::array<AtlasSubface<dim>, atlas_max_subfaces>();
            wells_.clear();
            trajectories_.clear();
            transfers_.clear();
            total_forward_inflow_ = 0.0;
            total_forward_outflow_ = 0.0;
        mass_balance_error_ = 0.0;
        maximum_origin_relative_error_ = 0.0;
        }

        void begin(const std::uint64_t cell_id,
                const unsigned int flow_step,
                const int tracking_direction,
                const TrajectoryAtlasOptions &options = TrajectoryAtlasOptions())
        {
            clear();
            AssertThrow(tracking_direction == -1 || tracking_direction == 1,
                        ExcMessage("Atlas tracking direction must be -1 or +1."));
            options_ = options;
            cell_id_ = cell_id;
            flow_step_ = flow_step;
            tracking_direction_ = tracking_direction;
            state_ = AtlasBuildState::collecting;
        }

        void set_subface(const AtlasSubface<dim> &subface)
        {
            require_collecting();
            AssertIndexRange(subface.face, atlas_faces_per_cell);
            AssertIndexRange(subface.slot, atlas_slots_per_face);
            AssertThrow(std::isfinite(subface.area) && subface.area >= 0.0,
                        ExcMessage("Atlas subface area must be finite and nonnegative."));
            AssertThrow(std::isfinite(subface.outward_flow),
                        ExcMessage("Atlas subface flow must be finite."));
            subfaces_[subface.id()] = subface;
        }

        void add_well(const AtlasWellTerminal &well)
        {
            require_collecting();
            AssertThrow(std::isfinite(well.outward_flow),
                        ExcMessage("Atlas well flow must be finite."));
            wells_.push_back(well);
        }

        void add_trajectory(const AtlasTrajectory<dim> &trajectory)
        {
            require_collecting();
            AssertThrow(trajectory.valid(),
                        dealii::ExcMessage("Invalid trajectory supplied to cell atlas."));
            if (trajectory.downstream_origin.kind == AtlasOriginKind::subface)
            {
                const unsigned int downstream = trajectory.downstream_origin.id;
                AssertThrow(subfaces_[downstream].active,
                            dealii::ExcMessage("Atlas trajectory starts on an inactive subface."));
                AssertThrow(
                    subfaces_[downstream].is_forward_outflow(options_.flow_tolerance),
                    dealii::ExcMessage("Backward atlas trajectory must start on a forward-outflow subface."));
            }
            else
            {
                AssertThrow(origin_well_flow(trajectory.downstream_origin.id) >
                                options_.flow_tolerance,
                            dealii::ExcMessage("Backward atlas trajectory must start at a forward aquifer outflow well."));
            }
            trajectories_.push_back(trajectory);
        }

        void finalize()
        {
            require_collecting();
            transfers_.clear();
            total_forward_inflow_ = 0.0;
            total_forward_outflow_ = 0.0;

            for (unsigned int f = 0; f < atlas_max_subfaces; ++f)
            {
                if (!subfaces_[f].active)
                    continue;
                total_forward_inflow_ += std::max(-subfaces_[f].outward_flow, 0.0);
                total_forward_outflow_ += std::max(subfaces_[f].outward_flow, 0.0);
            }
            for (unsigned int w = 0; w < wells_.size(); ++w)
            {
                total_forward_inflow_ += std::max(-wells_[w].outward_flow, 0.0);
                total_forward_outflow_ += std::max(wells_[w].outward_flow, 0.0);
            }

        typedef std::pair<AtlasOrigin, AtlasTerminal> TransferKey;
        std::map<TransferKey, double> flow_by_transfer;
        std::map<AtlasOrigin, double> represented_by_origin;
            for (unsigned int a = 0; a < trajectories_.size(); ++a)
            {
                const AtlasTrajectory<dim> &trajectory = trajectories_[a];
            flow_by_transfer[TransferKey(trajectory.downstream_origin,
                                         trajectory.upstream_terminal)] +=
                trajectory.represented_flow;
            represented_by_origin[trajectory.downstream_origin] +=
                trajectory.represented_flow;
            }

            for (typename std::map<TransferKey, double>::const_iterator it =
                    flow_by_transfer.begin();
                it != flow_by_transfer.end(); ++it)
            {
                AtlasTransferEntry entry;
                entry.downstream_origin = it->first.first;
                entry.upstream_terminal = it->first.second;
                entry.flow = it->second;
                const double row_flow = origin_outflow(entry.downstream_origin);
                entry.fraction = row_flow > options_.flow_tolerance
                                    ? entry.flow / row_flow
                                    : 0.0;
                transfers_.push_back(entry);
            }

        mass_balance_error_ = total_forward_outflow_ - total_forward_inflow_;
        maximum_origin_relative_error_ = 0.0;
        for (unsigned int f = 0; f < atlas_max_subfaces; ++f)
        {
            const double expected = std::max(subfaces_[f].outward_flow, 0.0);
            if (expected <= options_.flow_tolerance)
                continue;
            const AtlasOrigin origin = AtlasOrigin::subface(f);
            const double represented = represented_by_origin[origin];
            const double scale = std::max(expected, options_.flow_tolerance);
            maximum_origin_relative_error_ = std::max(
                maximum_origin_relative_error_,
                std::abs(represented - expected) / scale);
        }
        for (unsigned int w = 0; w < wells_.size(); ++w)
        {
            const double expected = std::max(wells_[w].outward_flow, 0.0);
            if (expected <= options_.flow_tolerance)
                continue;
            const AtlasOrigin origin = AtlasOrigin::well(wells_[w].well_id);
            const double represented = represented_by_origin[origin];
            const double scale = std::max(expected, options_.flow_tolerance);
            maximum_origin_relative_error_ = std::max(
                maximum_origin_relative_error_,
                std::abs(represented - expected) / scale);
        }
        state_ = AtlasBuildState::finalized;
        }

        AtlasQueryResult<dim> query(const Point<dim> &position,
                                    const double available_time) const
        {
            AssertThrow(state_ == AtlasBuildState::finalized,
                        dealii::ExcMessage("Trajectory atlas must be finalized before querying."));
            AssertThrow(std::isfinite(available_time) && available_time >= 0.0,
                        dealii::ExcMessage("Available atlas query time must be finite and nonnegative."));

            struct Candidate
            {
                double distance2;
                unsigned int trajectory;
                unsigned int sample;
            };

            std::vector<Candidate> candidates;
            for (unsigned int a = 0; a < trajectories_.size(); ++a)
                for (unsigned int s = 0; s < trajectories_[a].samples.size(); ++s)
                {
                    Candidate candidate;
                    candidate.distance2 = metric_distance_squared(
                        position, trajectories_[a].samples[s].position);
                    candidate.trajectory = a;
                    candidate.sample = s;
                    candidates.push_back(candidate);
                }

            AtlasQueryResult<dim> result;
            if (candidates.empty())
                return result;

            std::sort(candidates.begin(), candidates.end(),
                    [](const Candidate &a, const Candidate &b) {
                        return a.distance2 < b.distance2;
                    });
            if (options_.maximum_query_samples > 0 &&
                candidates.size() > options_.maximum_query_samples)
                candidates.resize(options_.maximum_query_samples);

            struct BranchAccumulator
            {
                double weight = 0.0;
                double remaining_time = 0.0;
                double remaining_length = 0.0;
                double advanced_time = 0.0;
                double advanced_length = 0.0;
                Point<dim> advanced_position;
            };

            std::map<AtlasTerminal, BranchAccumulator> accumulators;
            double total_kernel_weight = 0.0;

            for (unsigned int c = 0; c < candidates.size(); ++c)
            {
                const Candidate &candidate = candidates[c];
                const AtlasTrajectory<dim> &trajectory =
                    trajectories_[candidate.trajectory];
                const AtlasTrajectorySample<dim> &sample =
                    trajectory.samples[candidate.sample];

                const double control_time = sample_control_time(trajectory, candidate.sample);
                const double volume_weight =
                    trajectory.represented_flow * std::max(control_time, options_.flow_tolerance);
                const double kernel = std::pow(
                    candidate.distance2 +
                        options_.kernel_epsilon * options_.kernel_epsilon,
                    -0.5 * options_.kernel_power);
                const double weight = volume_weight * kernel;
                if (!(weight > 0.0) || !std::isfinite(weight))
                    continue;

                const double remaining_time =
                    std::max(trajectory.total_time() - sample.backward_time, 0.0);
                const double remaining_length =
                    std::max(trajectory.total_length() - sample.aquifer_length, 0.0);
                const double advance_time = std::min(available_time, remaining_time);
                const AtlasTrajectorySample<dim> advanced =
                    trajectory.sample_at_time(sample.backward_time + advance_time);

                BranchAccumulator &acc = accumulators[trajectory.upstream_terminal];
                acc.weight += weight;
                acc.remaining_time += weight * remaining_time;
                acc.remaining_length += weight * remaining_length;
                acc.advanced_time += weight * advance_time;
                acc.advanced_length +=
                    weight * std::max(advanced.aquifer_length - sample.aquifer_length, 0.0);
                for (unsigned int d = 0; d < dim; ++d)
                    acc.advanced_position[d] += weight * advanced.position[d];
                total_kernel_weight += weight;
            }

            if (!(total_kernel_weight > 0.0))
                return result;

            for (typename std::map<AtlasTerminal, BranchAccumulator>::const_iterator it =
                    accumulators.begin();
                it != accumulators.end(); ++it)
            {
                const BranchAccumulator &acc = it->second;
                const double raw_fraction = acc.weight / total_kernel_weight;
                if (raw_fraction < options_.minimum_branch_fraction)
                {
                    result.discarded_fraction += raw_fraction;
                    continue;
                }

                AtlasQueryBranch<dim> branch;
                branch.terminal = it->first;
                branch.fraction = raw_fraction;
                branch.remaining_time = acc.remaining_time / acc.weight;
                branch.remaining_length = acc.remaining_length / acc.weight;
                branch.advanced_time = acc.advanced_time / acc.weight;
                branch.advanced_length = acc.advanced_length / acc.weight;
                branch.reaches_terminal =
                    branch.remaining_time <= available_time + options_.flow_tolerance;
                for (unsigned int d = 0; d < dim; ++d)
                    branch.advanced_position[d] = acc.advanced_position[d] / acc.weight;

                result.retained_fraction += branch.fraction;
                result.branches.push_back(branch);
            }

            // Removed tiny branches are deliberately reported, then the retained
            // branches are normalized so packet splitting remains conservative.
            if (result.retained_fraction > 0.0)
                for (unsigned int b = 0; b < result.branches.size(); ++b)
                    result.branches[b].fraction /= result.retained_fraction;

            result.valid = !result.branches.empty();
            return result;
        }

        AtlasBuildState state() const { return state_; }
        bool is_finalized() const { return state_ == AtlasBuildState::finalized; }
        std::uint64_t cell_id() const { return cell_id_; }
        unsigned int flow_step() const { return flow_step_; }
        int tracking_direction() const { return tracking_direction_; }
        double total_forward_inflow() const { return total_forward_inflow_; }
        double total_forward_outflow() const { return total_forward_outflow_; }
    double mass_balance_error() const { return mass_balance_error_; }
    double maximum_origin_relative_error() const
    {
        return maximum_origin_relative_error_;
    }

        const TrajectoryAtlasOptions &options() const { return options_; }
        const std::array<AtlasSubface<dim>, atlas_max_subfaces> &subfaces() const
        {
            return subfaces_;
        }
        const std::vector<AtlasWellTerminal> &wells() const { return wells_; }
        const std::vector<AtlasTrajectory<dim> > &trajectories() const
        {
            return trajectories_;
        }
        const std::vector<AtlasTransferEntry> &transfers() const
        {
            return transfers_;
        }

    private:
        void require_collecting() const
        {
            AssertThrow(state_ == AtlasBuildState::collecting,
                        dealii::ExcMessage("Trajectory atlas is not in the collecting state."));
        }

        double metric_distance_squared(const dealii::Point<dim> &a,
                                    const dealii::Point<dim> &b) const
        {
            double value = 0.0;
            for (unsigned int d = 0; d < dim; ++d)
            {
                const double scale = options_.metric_scale[d];
                AssertThrow(std::isfinite(scale) && scale > 0.0,
                            dealii::ExcMessage("Atlas metric scales must be finite and positive."));
                const double dx = (a[d] - b[d]) / scale;
                value += dx * dx;
            }
            return value;
        }

        static double sample_control_time(const AtlasTrajectory<dim> &trajectory,
                                        const unsigned int sample)
        {
            const std::vector<AtlasTrajectorySample<dim> > &samples = trajectory.samples;
            if (samples.size() == 1)
                return 1.0;
            if (sample == 0)
                return 0.5 * (samples[1].backward_time - samples[0].backward_time);
            if (sample + 1 == samples.size())
                return 0.5 * (samples[sample].backward_time -
                            samples[sample - 1].backward_time);
            return 0.5 * (samples[sample + 1].backward_time -
                        samples[sample - 1].backward_time);
        }

        double origin_well_flow(const std::uint32_t well_id) const
        {
            for (unsigned int w = 0; w < wells_.size(); ++w)
                if (wells_[w].well_id == well_id)
                    return wells_[w].outward_flow;
            return 0.0;
        }

        double origin_outflow(const AtlasOrigin &origin) const
        {
            if (origin.kind == AtlasOriginKind::subface)
                return std::max(subfaces_[origin.id].outward_flow, 0.0);
            return std::max(origin_well_flow(origin.id), 0.0);
        }

        AtlasBuildState state_ = AtlasBuildState::empty;
        TrajectoryAtlasOptions options_;
        std::uint64_t cell_id_ = 0;
        unsigned int flow_step_ = 0;
        int tracking_direction_ = -1;

        std::array<AtlasSubface<dim>, atlas_max_subfaces> subfaces_;
        std::vector<AtlasWellTerminal> wells_;
        std::vector<AtlasTrajectory<dim> > trajectories_;
        std::vector<AtlasTransferEntry> transfers_;

        double total_forward_inflow_ = 0.0;
        double total_forward_outflow_ = 0.0;
    double mass_balance_error_ = 0.0;
    double maximum_origin_relative_error_ = 0.0;
};

} // namespace npsat_trace

#endif // NPSAT_V2_TRACE_TRAJ_ATLAS_H
