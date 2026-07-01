//
// Created by giorgk on 6/27/26.
//

#ifndef RT0_FACE_MAP_H
#define RT0_FACE_MAP_H

namespace npsat_trace {
    using namespace dealii;

    template <int dim>
    class RT0FaceMap {
    public:
        using global_dof_index = types::global_dof_index;
        static constexpr unsigned int faces_per_cell = GeometryInfo<dim>::faces_per_cell;

        using FaceGidArray      = std::array<global_dof_index, faces_per_cell>;
        using FaceSignArray     = std::array<std::int8_t,  faces_per_cell>;
        using FaceOwnedPosArray = std::array<std::int32_t, faces_per_cell>;
        using FaceFlagsArray    = std::array<std::uint8_t, faces_per_cell>;

        void clear()
        {
            owned_gids_ref.clear();
            face_gid.clear();
            face_sign.clear();
            face_owned_pos.clear();
            face_flags.clear();
        }

        void resize(const unsigned int n_local_cells)
        {
            face_gid.resize(n_local_cells);
            face_sign.resize(n_local_cells);
            face_owned_pos.resize(n_local_cells);
            face_flags.resize(n_local_cells);

            for (auto &a : face_owned_pos)
                for (auto &x : a)
                    x = -1;
        }

        unsigned int n_slots() const
        {
            return static_cast<unsigned int>(face_gid.size());
        }

        std::vector<global_dof_index> & owned_gids()
        {
            return owned_gids_ref;
        }

        const std::vector<global_dof_index> & owned_gids() const
        {
            return owned_gids_ref;
        }

        FaceGidArray & gids(const unsigned int slot)
        {
            AssertIndexRange(slot, face_gid.size());
            return face_gid[slot];
        }

        const FaceGidArray & gids(const unsigned int slot) const
        {
            AssertIndexRange(slot, face_gid.size());
            return face_gid[slot];
        }
        FaceSignArray & signs(const unsigned int slot)
        {
            AssertIndexRange(slot, face_sign.size());
            return face_sign[slot];
        }

        const FaceSignArray & signs(const unsigned int slot) const
        {
            AssertIndexRange(slot, face_sign.size());
            return face_sign[slot];
        }

        FaceOwnedPosArray & owned_positions(const unsigned int slot)
        {
            AssertIndexRange(slot, face_owned_pos.size());
            return face_owned_pos[slot];
        }

        const FaceOwnedPosArray & owned_positions(const unsigned int slot) const
        {
            AssertIndexRange(slot, face_owned_pos.size());
            return face_owned_pos[slot];
        }

        FaceFlagsArray & flags(const unsigned int slot)
        {
            AssertIndexRange(slot, face_flags.size());
            return face_flags[slot];
        }

        const FaceFlagsArray & flags(const unsigned int slot) const
        {
            AssertIndexRange(slot, face_flags.size());
            return face_flags[slot];
        }

        global_dof_index face_gid_at(const unsigned int slot,
                                     const unsigned int f) const
        {
            AssertIndexRange(slot, face_gid.size());
            AssertIndexRange(f, faces_per_cell);
            return face_gid[slot][f];
        }

        std::int8_t face_sign_at(const unsigned int slot,
                             const unsigned int f) const
        {
            AssertIndexRange(slot, face_sign.size());
            AssertIndexRange(f, faces_per_cell);
            return face_sign[slot][f];
        }

        std::int32_t face_owned_pos_at(const unsigned int slot,
                                   const unsigned int f) const
        {
            AssertIndexRange(slot, face_owned_pos.size());
            AssertIndexRange(f, faces_per_cell);
            return face_owned_pos[slot][f];
        }

        std::uint8_t face_flags_at(const unsigned int slot,
                                   const unsigned int f) const
        {
            AssertIndexRange(slot, face_flags.size());
            AssertIndexRange(f, faces_per_cell);
            return face_flags[slot][f];
        }

        bool is_coarse_parent_face(const unsigned int slot,
                               const unsigned int f) const
        {
            return (face_flags_at(slot, f) & std::uint8_t(0x1)) != 0;
        }

        bool is_refined_child_face(const unsigned int slot,
                                   const unsigned int f) const
        {
            return (face_flags_at(slot, f) & std::uint8_t(0x2)) != 0;
        }

        bool is_regular_face(const unsigned int slot,
                             const unsigned int f) const
        {
            return !is_coarse_parent_face(slot, f) &&
                   !is_refined_child_face(slot, f);
        }

        double vn_outward(const unsigned int slot,
                      const unsigned int f,
                      const dealii::TrilinosWrappers::MPI::Vector &vface) const
        {
            const auto gid = face_gid_at(slot, f);
            const double s = static_cast<double>(face_sign_at(slot, f));
            return s * vface[gid];
        }
    private:
        std::vector<global_dof_index> owned_gids_ref;
        std::vector<FaceGidArray>      face_gid;
        std::vector<FaceSignArray>     face_sign;
        std::vector<FaceOwnedPosArray> face_owned_pos;
        std::vector<FaceFlagsArray>    face_flags;
    };

}

#endif //RT0_FACE_MAP_H
