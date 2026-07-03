//
// Created by giorgk on 6/27/26.
//

#ifndef CACHED_VELOCITY_H
#define CACHED_VELOCITY_H
namespace npsat_trace {
    using namespace dealii;

    // ------------------------------------------------------------
    // Face order used everywhere in this class:
    //
    //   0 = -x
    //   1 = +x
    //   2 = -y
    //   3 = +y
    //   4 = -z
    //   5 = +z
    //
    // All stored face-normal velocities are OUTWARD with respect to
    // the cell or subcell to which they belong.
    // ------------------------------------------------------------
    enum FaceId : unsigned int { xm=0, xp=1, ym=2, yp=3, zm=4, zp=5 };

    template<int dim>
    static unsigned int opposite_face(const unsigned int f)
    {
        AssertIndexRange(f, GeometryInfo<dim>::faces_per_cell);
        return f ^ 1;
    }

    static unsigned int canonical_slot_from_projected_offsets(const double a,
                                                          const double b)
    {
        const unsigned int ia = (a >= 0.0 ? 1u : 0u);
        const unsigned int ib = (b >= 0.0 ? 1u : 0u);
        return ia + 2u * ib;
    }

    // ------------------------------------------------------------
    // Subcell ordering:
    //
    //   s = ix + 2*iy + 4*iz
    //
    // where
    //   ix = 0 -> -x half,  ix = 1 -> +x half
    //   iy = 0 -> -y half,  iy = 1 -> +y half
    //   iz = 0 -> -z half,  iz = 1 -> +z half
    //
    // Therefore:
    //   s=0 : (-x,-y,-z)
    //   s=1 : (+x,-y,-z)
    //   s=2 : (-x,+y,-z)
    //   s=3 : (+x,+y,-z)
    //   s=4 : (-x,-y,+z)
    //   s=5 : (+x,-y,+z)
    //   s=6 : (-x,+y,+z)
    //   s=7 : (+x,+y,+z)
    // ------------------------------------------------------------
    struct SubcellRT0Data
    {
        // Outward normal velocity on the 6 faces of this subcell
        // face order: xm, xp, ym, yp, zm, zp
        std::array<double,6> vn_outward{};
    };

    /**
         * Return the canonical index of one of the 8 split RT0 subcells.
         *
         * The parent cell is split into a 2x2x2 arrangement:
         *
         *   ix in {0,1} : negative-x half / positive-x half
         *   iy in {0,1} : negative-y half / positive-y half
         *   iz in {0,1} : negative-z half / positive-z half
         *
         * Indexing is:
         *
         *   s = ix + 2*iy + 4*iz
         *
         * Therefore:
         *
         *   s = 0 -> (ix,iy,iz) = (0,0,0)
         *   s = 1 -> (1,0,0)
         *   s = 2 -> (0,1,0)
         *   s = 3 -> (1,1,0)
         *   s = 4 -> (0,0,1)
         *   s = 5 -> (1,0,1)
         *   s = 6 -> (0,1,1)
         *   s = 7 -> (1,1,1)
         *
         * This is consistent with the face-slot indexing used by:
         *
         *   q_yz(iy,iz) for x-faces
         *   q_xz(ix,iz) for y-faces
         *   q_xy(ix,iy) for z-faces
         *
         * so that each outer parent-face subface maps naturally to the
         * corresponding split subcell.
         */
    static unsigned int subcell_index(const unsigned int ix,
                                  const unsigned int iy,
                                  const unsigned int iz)
    {
        return ix + 2 * iy + 4 * iz;
    }

    // On x-faces, the four subfaces are indexed by (iy,iz)
    static unsigned int q_yz(const unsigned int iy, const unsigned int iz)
    {
        return iy + 2 * iz;
    }

    // On y-faces, the four subfaces are indexed by (ix,iz)
    static unsigned int q_xz(const unsigned int ix, const unsigned int iz)
    {
        return ix + 2 * iz;
    }

    // On z-faces, the four subfaces are indexed by (ix,iy)
    static unsigned int q_xy(const unsigned int ix, const unsigned int iy)
    {
        return ix + 2 * iy;
    }

    double opposite_subface_to_inner(const double v1, const double v2)
    {
        // Returns the inner-face normal velocity signed with respect to
        // the subcell of the first argument.
        //
        // Convention:
        //   v1 = outward velocity on subcell 1 outer face
        //   v2 = outward velocity on opposite subcell 2 outer face
        //
        // Returned value:
        //   > 0 : flow is outward from subcell 1 across the inner face
        //         (i.e. from subcell 1 toward subcell 2)
        //   < 0 : flow is inward to subcell 1 across the inner face
        //         (i.e. from subcell 2 toward subcell 1)
        //
        // Reason:
        //   v1 must be reversed to express it in the "1 -> 2" direction,
        //   while v2 already has that orientation.
        //
        //   inner = 0.5 * [(-v1) + (v2)] = 0.5 * (v2 - v1)

        return 0.5 * (v2 - v1);
    }


    template<int dim>
    class CellVelocityCacheRT0Split3D {
    public:
        static_assert(dim == 3, "This cache is designed for 3D.");

        using CellIt = typename DoFHandler<dim>::active_cell_iterator;

        void init_cache(const CellIt &cell_in, const RT0FaceMap<dim> &rt0_map,
            const TrilinosWrappers::MPI::Vector &vface);

        const std::array<double, 4> &get_xv() const { return xv; }
        const std::array<double, 4> &get_yv() const { return yv; }
        const std::array<double, 4> &get_zb() const { return zb; }
        const std::array<double, 4> &get_zt() const { return zt; }

        void clear();
        void get_clamped_ref_coords(const Point<dim> &p, Point<dim> &p_ref);
        void compute_velocity_at_particle(const Point<dim> &x_ref, Tensor<1,dim> &u_phys, double &vmag_out) const;


    private:
        std::array<double,4> build_face_subface_values_canonical(const unsigned int f,const RT0FaceMap<dim> &rt0_map,
            const TrilinosWrappers::MPI::Vector &vface) const;

        void build_cell_ordering_directions(Tensor<1,dim> &ex_dir, Tensor<1,dim> &ey_dir, Tensor<1,dim> &ez_dir) const;
        void get_face_ordering_directions(const unsigned int f,
            const Tensor<1,dim> &ex_dir, const Tensor<1,dim> &ey_dir, const Tensor<1,dim> &ez_dir,
            Tensor<1,dim> &dir1, Tensor<1,dim> &dir2) const;
        void build_subcells();
        void write_subcells_arrays_to_txt(const std::string &filename) const;
        Tensor<1, dim> interpolate_rt0_reference_velocity(const Point<dim> &p_ref) const;
        void locate_subcell_and_local_coords(const Point<dim> &p_ref, unsigned int &subcell_id_out,
                                            double &rx, double &ry, double &rz) const;
        Tensor<1, dim> interpolate_rt0_on_subcell(const unsigned int s, const double rx, const double ry, const double rz) const;

        bool compute_reference_point(const Point<3> &x_phys,Point<3> &x_ref) const;
        Point<dim> map_reference_point_to_physical(const Point<dim> &x_ref, bool space01 = false) const;


        CellIt cell;

        // ------------------------------------------------------------
        // For each parent face, store 4 subface-normal velocities.
        // All are outward with respect to the PARENT cell.
        //
        // Face order:
        //   0=-x, 1=+x, 2=-y, 3=+y, 4=-z, 5=+z
        //
        // Subface indexing:
        //
        //   on x-faces: q = iy + 2*iz
        //   on y-faces: q = ix + 2*iz
        //   on z-faces: q = ix + 2*iy
        // ------------------------------------------------------------
        std::array<std::array<double, 4>, 6> face_subface_vn{};

        // 8 RT0 subcells
        std::array<SubcellRT0Data, 8> subcells{};
        std::array<double, 4> xv{}, yv{};
        std::array<double, 4> zb{}, zt{};

        static constexpr unsigned int perm_ccw[4] = {0, 1, 3, 2};
        const unsigned int f_bot = 4; // -z
        const unsigned int f_top = 5; // +z


    };

    template <int dim>
    constexpr unsigned int CellVelocityCacheRT0Split3D<dim>::perm_ccw[4];

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::init_cache(const CellIt &cell_in, const RT0FaceMap<dim> &rt0_map,
        const TrilinosWrappers::MPI::Vector &vface) {

        static_assert(std::is_same<CellIt, typename DoFHandler<dim>::active_cell_iterator>::value,
            "CellIt must be DoFHandler<dim>::active_cell_iterator");

        clear();

        cell = cell_in;
        std::cout << cell->id() << std::endl;
        AssertThrow(cell->is_active(), dealii::ExcMessage("Expected active cell."));
        AssertThrow(cell->is_locally_owned(), dealii::ExcMessage("Expected locally owned cell."));

        //const unsigned int slot = static_cast<unsigned int>(cell->user_index());

        // Step 1: build canonical 2x2 face-subface values for all 6 parent faces
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f) {
            //std::cout << "Face " << f << ": ";
            face_subface_vn[f] = build_face_subface_values_canonical(f, rt0_map, vface);
            // for (unsigned int iv=0; iv < face_subface_vn[f].size(); ++iv)
            //     std::cout << face_subface_vn[f][iv] << " ";
            // std::cout << std::endl;
        }

        // ------------------------------------------------------------
        // Build the 8 RT0 subcells from the outer face subface values
        // and from the derived inner face values.
        // ------------------------------------------------------------
        build_subcells();

        // ------------------------------------------------------------
        // Cache geometry of the extruded hexahedron.
        // The bottom and top faces are assumed to have matching XY.
        // ------------------------------------------------------------
        for (unsigned int k = 0; k < 4; ++k) {
            const unsigned int i = perm_ccw[k];

            const auto pb = cell->face(f_bot)->vertex(i);
            const auto pt = cell->face(f_top)->vertex(i);

            xv[i] = pb[0];
            yv[i] = pb[1];

            // consistency check (extruded): top and bottom x,y should match
            // (You can relax tolerance if needed)
            AssertThrow(std::abs(pt[0] - pb[0]) < 1e-10 && std::abs(pt[1] - pb[1]) < 1e-10,
                        dealii::ExcMessage("Mesh is not strictly extruded: top/bottom XY mismatch."));

            zb[i] = pb[2];
            zt[i] = pt[2];
        }

        {
            const std::string fn = "init_cell_" + cell->id().to_string();
            std::cout << "Writing subcells to " << fn << std::endl;
            write_subcells_arrays_to_txt(fn);
        }

    }

    template<int dim>
    std::array<double,4>CellVelocityCacheRT0Split3D<dim>::build_face_subface_values_canonical(
        const unsigned int f,const RT0FaceMap<dim> &rt0_map, const TrilinosWrappers::MPI::Vector &vface) const {
        std::array<double,4> vsub{};
        vsub.fill(0.0);

        const unsigned int slot = static_cast<unsigned int>(cell->user_index());

        // Regular face: no subfaces on this side.
        // Use the single RT0 face value and replicate to all 4 canonical slots.
        if (!cell->face(f)->has_children()) {
            const double v = rt0_map.vn_outward(slot, f, vface);
            vsub.fill(v);
            return vsub;
        }

        // Coarse-side parent face at a coarse-fine interface.
        const unsigned int n_sub = cell->face(f)->n_children();
        AssertThrow(n_sub == 4, ExcMessage("Expected exactly 4 subfaces on refined face."));

        //const auto canonical_centers = build_parent_face_canonical_centers(f);
        // Build once per coarse-fine interface
        Tensor<1,dim> ex_dir, ey_dir, ez_dir;
        build_cell_ordering_directions(ex_dir, ey_dir, ez_dir);
        Tensor<1,dim> dir1, dir2;
        get_face_ordering_directions(f, ex_dir, ey_dir, ez_dir, dir1, dir2);

        const Point<dim> cf = cell->face(f)->center();

        for (unsigned int subface = 0; subface < n_sub; ++subface) {
            const auto coarse_subface = cell->face(f)->child(subface);
            // Active refined neighbor cell touching this subface
            const auto neighbor_child = cell->neighbor_child_on_subface(f, subface);
            AssertThrow(neighbor_child->is_active(), ExcMessage("Expected active neighbor child on subface."));

            // Find which face of neighbor_child matches coarse_subface
            const unsigned int nchild_face = opposite_face<dim>(f);
            const unsigned int child_slot = static_cast<unsigned int>(neighbor_child->user_index());

            // RT0 velocity outward with respect to the refined child cell
            const double v_child_outward = rt0_map.vn_outward(child_slot, nchild_face, vface);

            // Convert to outward with respect to THIS coarse cell
            const double v_coarse_outward = -v_child_outward;

            // Match this actual child subface center to the canonical parent-face slot
            const Point<dim> child_center = coarse_subface->center();
            const Tensor<1,dim> d = child_center - cf;
            const double a = d * dir1;
            const double b = d * dir2;
            const unsigned int q = canonical_slot_from_projected_offsets(a, b);

            vsub[q] = v_coarse_outward;
        }
        return vsub;
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::build_cell_ordering_directions(
        Tensor<1,dim> &ex_dir, Tensor<1,dim> &ey_dir, Tensor<1,dim> &ez_dir) const {
        const Point<dim> c_xm = cell->face(xm)->center();
        const Point<dim> c_xp = cell->face(xp)->center();
        const Point<dim> c_ym = cell->face(ym)->center();
        const Point<dim> c_yp = cell->face(yp)->center();
        const Point<dim> c_zm = cell->face(zm)->center();
        const Point<dim> c_zp = cell->face(zp)->center();

        ex_dir = c_xp - c_xm;
        ey_dir = c_yp - c_ym;
        ez_dir = c_zp - c_zm;

        const double nx = ex_dir.norm();
        const double ny = ey_dir.norm();
        const double nz = ez_dir.norm();

        AssertThrow(nx > 0.0 && ny > 0.0 && nz > 0.0, ExcMessage("Degenerate cell ordering directions."));

        ex_dir /= nx;
        ey_dir /= ny;
        ez_dir /= nz;
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::get_face_ordering_directions(const unsigned int f,
            const Tensor<1,dim> &ex_dir, const Tensor<1,dim> &ey_dir, const Tensor<1,dim> &ez_dir,
            Tensor<1,dim> &dir1, Tensor<1,dim> &dir2) const {
        switch (f)
        {
            case xm:
            case xp:
                dir1 = ey_dir;
            dir2 = ez_dir;
            break;

            case ym:
            case yp:
                dir1 = ex_dir;
            dir2 = ez_dir;
            break;

            case zm:
            case zp:
                dir1 = ex_dir;
            dir2 = ey_dir;
            break;

            default:
                AssertThrow(false, ExcInternalError());
        }
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::build_subcells() {
        for (auto &sc : subcells)
            sc.vn_outward.fill(0.0);

        // --------------------------------------------------------
        // 1. Assign outer faces
        // --------------------------------------------------------
        for (unsigned int iz = 0; iz < 2; ++iz) {
            for (unsigned int iy = 0; iy < 2; ++iy) {
                for (unsigned int ix = 0; ix < 2; ++ix) {
                    const unsigned int s = subcell_index(ix, iy, iz);

                    // x outer face: indexed on yz
                    if (ix == 0)
                        subcells[s].vn_outward[xm] = face_subface_vn[xm][q_yz(iy, iz)];
                    else
                        subcells[s].vn_outward[xp] = face_subface_vn[xp][q_yz(iy, iz)];

                    // y outer face: indexed on xz
                    if (iy == 0)
                        subcells[s].vn_outward[ym] = face_subface_vn[ym][q_xz(ix, iz)];
                    else
                        subcells[s].vn_outward[yp] = face_subface_vn[yp][q_xz(ix, iz)];

                    // z outer face: indexed on xy
                    if (iz == 0)
                        subcells[s].vn_outward[zm] = face_subface_vn[zm][q_xy(ix, iy)];
                    else
                        subcells[s].vn_outward[zp] = face_subface_vn[zp][q_xy(ix, iy)];
                }
            }
        }

        // --------------------------------------------------------
        // 2. Assign inner x-faces
        //
        // Built from the -x subcell, then copied by negation to the
        // adjacent +x subcell.
        // --------------------------------------------------------
        for (unsigned int iz = 0; iz < 2; ++iz) {
            for (unsigned int iy = 0; iy < 2; ++iy) {
                const unsigned int q = q_yz(iy, iz);

                const double vxm = face_subface_vn[xm][q];
                const double vxp = face_subface_vn[xp][q];

                const double vx_inner = opposite_subface_to_inner(vxm, vxp);

                const unsigned int s_left  = subcell_index(0, iy, iz);
                const unsigned int s_right = subcell_index(1, iy, iz);

                subcells[s_left].vn_outward[xp]   = +vx_inner;
                subcells[s_right].vn_outward[xm]  = -vx_inner;
            }
        }

        // --------------------------------------------------------
        // 3. Assign inner y-faces
        //
        // Built from the -y subcell, then copied by negation to the
        // adjacent +y subcell.
        // --------------------------------------------------------
        for (unsigned int iz = 0; iz < 2; ++iz) {
            for (unsigned int ix = 0; ix < 2; ++ix) {
                const unsigned int q = q_xz(ix, iz);

                const double vym = face_subface_vn[ym][q];
                const double vyp = face_subface_vn[yp][q];

                const double vy_inner = opposite_subface_to_inner(vym, vyp);

                const unsigned int s_back  = subcell_index(ix, 0, iz);
                const unsigned int s_front = subcell_index(ix, 1, iz);

                subcells[s_back].vn_outward[yp]   = +vy_inner;
                subcells[s_front].vn_outward[ym]  = -vy_inner;
            }
        }

        // --------------------------------------------------------
        // 4. Assign inner z-faces
        //
        // Built from the -z subcell, then copied by negation to the
        // adjacent +z subcell.
        // --------------------------------------------------------
        for (unsigned int iy = 0; iy < 2; ++iy) {
            for (unsigned int ix = 0; ix < 2; ++ix) {
                const unsigned int q = q_xy(ix, iy);

                const double vzm = face_subface_vn[zm][q];
                const double vzp = face_subface_vn[zp][q];

                const double vz_inner = opposite_subface_to_inner(vzm, vzp);

                const unsigned int s_bottom = subcell_index(ix, iy, 0);
                const unsigned int s_top    = subcell_index(ix, iy, 1);

                subcells[s_bottom].vn_outward[zp] = +vz_inner;
                subcells[s_top].vn_outward[zm]    = -vz_inner;
            }
        }
    }


    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::write_subcells_arrays_to_txt(const std::string &filename) const {
        auto open_out = [](const std::string &fn) -> std::ofstream {
            std::ofstream out(fn);
            if (!out)
                throw std::runtime_error("Failed to open file for writing: " + fn);
            out << std::setprecision(17);
            return out;
        };

        auto check_out = [](std::ofstream &out, const std::string &fn) {
            if (!out)
                throw std::runtime_error("Failed while writing file: " + fn);
        };

        auto ref01_to_phys = [&](const double x, const double y, const double z) -> Point<dim> {
            Point<dim> p_ref;
            p_ref[0] = x;
            p_ref[1] = y;
            p_ref[2] = z;
            return map_reference_point_to_physical(p_ref, true);
        };

        // ------------------------------------------------------------
        // 1. subcells_vn_outward  (8 x 6)
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_subcells_vn_outward.txt";
            std::ofstream out = open_out(fn);

            out << "# subcells_vn_outward\n";
            out << "# Face order: [xm, xp, ym, yp, zm, zp]\n";
            out << "# Rows correspond to subcell s = ix + 2*iy + 4*iz\n";
            out << "# Columns: xm xp ym yp zm zp\n";

            for (unsigned int s = 0; s < 8; ++s)
            {
                const auto &vn = subcells[s].vn_outward;
                out << vn[xm] << " "
                    << vn[xp] << " "
                    << vn[ym] << " "
                    << vn[yp] << " "
                    << vn[zm] << " "
                    << vn[zp] << "\n";
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 2. face_subface_vn  (6 x 4)
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_face_subface_vn.txt";
            std::ofstream out = open_out(fn);

            out << "# face_subface_vn\n";
            out << "# Face order: [xm, xp, ym, yp, zm, zp]\n";
            out << "# Each row contains the 4 subface-normal velocities for one face\n";
            out << "# Columns: q0 q1 q2 q3\n";

            for (unsigned int f = 0; f < 6; ++f)
            {
                out << face_subface_vn[f][0] << " "
                    << face_subface_vn[f][1] << " "
                    << face_subface_vn[f][2] << " "
                    << face_subface_vn[f][3] << "\n";
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 3. subcell_centers in reference space (8 x 3)
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_subcell_centers.txt";
            std::ofstream out = open_out(fn);

            out << "# subcell_centers\n";
            out << "# Rows correspond to subcell s = ix + 2*iy + 4*iz\n";
            out << "# Columns: x y z\n";

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    for (unsigned int ix = 0; ix < 2; ++ix)
                    {
                        const double cx = (ix == 0 ? 0.25 : 0.75);
                        const double cy = (iy == 0 ? 0.25 : 0.75);
                        const double cz = (iz == 0 ? 0.25 : 0.75);

                        out << cx << " " << cy << " " << cz << "\n";
                    }
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 3b. subcell_centers in physical space (8 x 3)
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_subcell_centers_phys.txt";
            std::ofstream out = open_out(fn);

            out << "# subcell_centers in physical space\n";
            out << "# Rows correspond to subcell s = ix + 2*iy + 4*iz\n";
            out << "# Columns: X Y Z\n";

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    for (unsigned int ix = 0; ix < 2; ++ix)
                    {
                        const double cx = (ix == 0 ? 0.25 : 0.75);
                        const double cy = (iy == 0 ? 0.25 : 0.75);
                        const double cz = (iz == 0 ? 0.25 : 0.75);

                        const Point<dim> p = ref01_to_phys(cx, cy, cz);
                        out << p[0] << " " << p[1] << " " << p[2] << "\n";
                    }
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 4. outer_face_centers  (24 x 6)
        // Columns: x y z vx vy vz
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_outer_face_centers.txt";
            std::ofstream out = open_out(fn);

            out << "# outer_face_centers\n";
            out << "# Columns: x y z vx vy vz\n";

            // xm/xp : q_yz(iy,iz)
            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    const unsigned int q = q_yz(iy, iz);
                    const double y = (iy == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    out << 0.0 << " " << y << " " << z << " "
                        << (-face_subface_vn[xm][q]) << " " << 0.0 << " " << 0.0 << "\n";

                    out << 1.0 << " " << y << " " << z << " "
                        << ( face_subface_vn[xp][q]) << " " << 0.0 << " " << 0.0 << "\n";
                }
            }

            // ym/yp : q_xz(ix,iz)
            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int q = q_xz(ix, iz);
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    out << x << " " << 0.0 << " " << z << " "
                        << 0.0 << " " << (-face_subface_vn[ym][q]) << " " << 0.0 << "\n";

                    out << x << " " << 1.0 << " " << z << " "
                        << 0.0 << " " << ( face_subface_vn[yp][q]) << " " << 0.0 << "\n";
                }
            }

            // zm/zp : q_xy(ix,iy)
            for (unsigned int iy = 0; iy < 2; ++iy)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int q = q_xy(ix, iy);
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double y = (iy == 0 ? 0.25 : 0.75);

                    out << x << " " << y << " " << 0.0 << " "
                        << 0.0 << " " << 0.0 << " " << (-face_subface_vn[zm][q]) << "\n";

                    out << x << " " << y << " " << 1.0 << " "
                        << 0.0 << " " << 0.0 << " " << ( face_subface_vn[zp][q]) << "\n";
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 4b. outer_face_centers in physical space (24 x 6)
        // Columns: X Y Z vx vy vz
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_outer_face_centers_phys.txt";
            std::ofstream out = open_out(fn);

            out << "# outer_face_centers in physical space\n";
            out << "# Columns: X Y Z vx vy vz\n";

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    const unsigned int q = q_yz(iy, iz);
                    const double y = (iy == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    {
                        const Point<dim> p = ref01_to_phys(0.0, y, z);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << (-face_subface_vn[xm][q]) << " " << 0.0 << " " << 0.0 << "\n";
                    }
                    {
                        const Point<dim> p = ref01_to_phys(1.0, y, z);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << ( face_subface_vn[xp][q]) << " " << 0.0 << " " << 0.0 << "\n";
                    }
                }
            }

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int q = q_xz(ix, iz);
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    {
                        const Point<dim> p = ref01_to_phys(x, 0.0, z);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << 0.0 << " " << (-face_subface_vn[ym][q]) << " " << 0.0 << "\n";
                    }
                    {
                        const Point<dim> p = ref01_to_phys(x, 1.0, z);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << 0.0 << " " << ( face_subface_vn[yp][q]) << " " << 0.0 << "\n";
                    }
                }
            }

            for (unsigned int iy = 0; iy < 2; ++iy)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int q = q_xy(ix, iy);
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double y = (iy == 0 ? 0.25 : 0.75);

                    {
                        const Point<dim> p = ref01_to_phys(x, y, 0.0);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << 0.0 << " " << 0.0 << " " << (-face_subface_vn[zm][q]) << "\n";
                    }
                    {
                        const Point<dim> p = ref01_to_phys(x, y, 1.0);
                        out << p[0] << " " << p[1] << " " << p[2] << " "
                            << 0.0 << " " << 0.0 << " " << ( face_subface_vn[zp][q]) << "\n";
                    }
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 5. inner_face_centers  (12 x 6)
        // Columns: x y z vx vy vz
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_inner_face_centers.txt";
            std::ofstream out = open_out(fn);

            out << "# inner_face_centers\n";
            out << "# Columns: x y z vx vy vz\n";

            // x-inner: use left subcell xp
            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    const unsigned int s_left = subcell_index(0, iy, iz);
                    const double v = subcells[s_left].vn_outward[xp];
                    const double y = (iy == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    out << 0.5 << " " << y << " " << z << " "
                        << v << " " << 0.0 << " " << 0.0 << "\n";
                }
            }

            // y-inner: use back subcell yp
            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int s_back = subcell_index(ix, 0, iz);
                    const double v = subcells[s_back].vn_outward[yp];
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    out << x << " " << 0.5 << " " << z << " "
                        << 0.0 << " " << v << " " << 0.0 << "\n";
                }
            }

            // z-inner: use bottom subcell zp
            for (unsigned int iy = 0; iy < 2; ++iy)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int s_bottom = subcell_index(ix, iy, 0);
                    const double v = subcells[s_bottom].vn_outward[zp];
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double y = (iy == 0 ? 0.25 : 0.75);

                    out << x << " " << y << " " << 0.5 << " "
                        << 0.0 << " " << 0.0 << " " << v << "\n";
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 5b. inner_face_centers in physical space (12 x 6)
        // Columns: X Y Z vx vy vz
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_inner_face_centers_phys.txt";
            std::ofstream out = open_out(fn);

            out << "# inner_face_centers in physical space\n";
            out << "# Columns: X Y Z vx vy vz\n";

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    const unsigned int s_left = subcell_index(0, iy, iz);
                    const double v = subcells[s_left].vn_outward[xp];
                    const double y = (iy == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    const Point<dim> p = ref01_to_phys(0.5, y, z);
                    out << p[0] << " " << p[1] << " " << p[2] << " "
                        << v << " " << 0.0 << " " << 0.0 << "\n";
                }
            }

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int s_back = subcell_index(ix, 0, iz);
                    const double v = subcells[s_back].vn_outward[yp];
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double z = (iz == 0 ? 0.25 : 0.75);

                    const Point<dim> p = ref01_to_phys(x, 0.5, z);
                    out << p[0] << " " << p[1] << " " << p[2] << " "
                        << 0.0 << " " << v << " " << 0.0 << "\n";
                }
            }

            for (unsigned int iy = 0; iy < 2; ++iy)
            {
                for (unsigned int ix = 0; ix < 2; ++ix)
                {
                    const unsigned int s_bottom = subcell_index(ix, iy, 0);
                    const double v = subcells[s_bottom].vn_outward[zp];
                    const double x = (ix == 0 ? 0.25 : 0.75);
                    const double y = (iy == 0 ? 0.25 : 0.75);

                    const Point<dim> p = ref01_to_phys(x, y, 0.5);
                    out << p[0] << " " << p[1] << " " << p[2] << " "
                        << 0.0 << " " << 0.0 << " " << v << "\n";
                }
            }

            check_out(out, fn);
        }

        // ------------------------------------------------------------
        // 6. cell wireframe corners in physical space (8 x 3)
        // Corner order in [0,1]^3:
        // 0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)
        // 4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)
        // ------------------------------------------------------------
        {
            const std::string fn = filename + "_cell_corners_phys.txt";
            std::ofstream out = open_out(fn);
            out << "# cell_corners_phys\n";
            out << "# Columns: X Y Z\n";
            out << "# Row order:\n";
            out << "# 0:(0,0,0)  1:(1,0,0)  2:(0,1,0)  3:(1,1,0)\n";
            out << "# 4:(0,0,1)  5:(1,0,1)  6:(0,1,1)  7:(1,1,1)\n";

            for (unsigned int iz = 0; iz < 2; ++iz)
            {
                for (unsigned int iy = 0; iy < 2; ++iy)
                {
                    for (unsigned int ix = 0; ix < 2; ++ix)
                    {
                        Point<dim> p_ref;
                        p_ref[0] = static_cast<double>(ix);
                        p_ref[1] = static_cast<double>(iy);
                        p_ref[2] = static_cast<double>(iz);

                        const Point<dim> p_phys = map_reference_point_to_physical(p_ref, true);

                        out << p_phys[0] << " "
                            << p_phys[1] << " "
                            << p_phys[2] << "\n";
                    }
                }
            }

            check_out(out, fn);
        }
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::get_clamped_ref_coords(const Point<dim> &p, Point<dim> &p_ref) {
        const bool ok = compute_reference_point(p,p_ref);
        AssertThrow(ok, dealii::ExcMessage("Failed to compute reference coordinates."));

        // This function should be called only if we are sure that point p is inside the cell.
        // However, because point inside uses linear mapping we clamp here the values
        // so that the point can be calculated. This is not going to add too much error for our purpose
        const double eps_in = 1e-10;
        for (unsigned int d = 0; d < dim; ++d)
            p_ref[d] = clamp_(p_ref[d], -1.0 + eps_in, 1.0 - eps_in);
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::compute_velocity_at_particle(const Point<dim> &x_ref, Tensor<1,dim> &u_phys, double &vmag_out) const {
        u_phys = interpolate_rt0_reference_velocity(x_ref);
        vmag_out = u_phys.norm();
    }

    /**
     * Interpolate the split RT0 velocity at a parent-cell reference point.
     *
     * Input:
     *   p_ref : point in the parent reference cell [-1,1]^3
     *
     * Steps:
     *   1. Determine which of the 8 split subcells contains p_ref.
     *   2. Map p_ref to local subcell coordinates (rx,ry,rz) in [0,1]^3.
     *   3. Evaluate the RT0 field on that subcell from its six outward
     *      face-normal velocities.
     *
     * Notes:
     *   - The split subcells are indexed with
     *
     *         s = ix + 2*iy + 4*iz
     *
     *     where ix,iy,iz ∈ {0,1}.
     *
     *   - On the internal split planes x=0, y=0, z=0 in parent reference space,
     *     the point is assigned to the positive-side subcell by convention
     *     because (p_ref[d] < 0 ? 0 : 1) is used.
     */
    template<int dim>
    Tensor<1, dim> CellVelocityCacheRT0Split3D<dim>::interpolate_rt0_reference_velocity(const Point<dim> &p_ref) const {
        unsigned int s = 0;
        double rx = 0.0, ry = 0.0, rz = 0.0;
        locate_subcell_and_local_coords(p_ref, s, rx, ry, rz);
        return interpolate_rt0_on_subcell(s, rx, ry, rz);
    }

    template<int dim>
    void CellVelocityCacheRT0Split3D<dim>::locate_subcell_and_local_coords(const Point<dim> &p_ref,
        unsigned int &subcell_id_out, double &rx, double &ry, double &rz) const {
        AssertThrow(p_ref[0] >= -1.0 - 1e-12 && p_ref[0] <= 1.0 + 1e-12 &&
                            p_ref[1] >= -1.0 - 1e-12 && p_ref[1] <= 1.0 + 1e-12 &&
                            p_ref[2] >= -1.0 - 1e-12 && p_ref[2] <= 1.0 + 1e-12,
                        dealii::ExcMessage("Reference point must lie in [-1,1]^3."));

        const unsigned int ix = (p_ref[0] < 0.0 ? 0U : 1U);
        const unsigned int iy = (p_ref[1] < 0.0 ? 0U : 1U);
        const unsigned int iz = (p_ref[2] < 0.0 ? 0U : 1U);

        subcell_id_out = subcell_index(ix, iy, iz);

        // Map parent reference coordinate to local [0,1] coordinate in the subcell.
        //
        // Parent interval [-1,0] -> local [0,1] for negative half
        // Parent interval [ 0,1] -> local [0,1] for positive half
        rx = (ix == 0 ? (p_ref[0] + 1.0) : p_ref[0]);
        ry = (iy == 0 ? (p_ref[1] + 1.0) : p_ref[1]);
        rz = (iz == 0 ? (p_ref[2] + 1.0) : p_ref[2]);

        // Clamp tiny roundoff excursions
        rx = std::max(0.0, std::min(1.0, rx));
        ry = std::max(0.0, std::min(1.0, ry));
        rz = std::max(0.0, std::min(1.0, rz));

        bool print_debug = false;
        if (print_debug)
        {
            std::cout << "p_ref = "
                      << p_ref[0] << " "
                      << p_ref[1] << " "
                      << p_ref[2] << std::endl;

            std::cout << "ix iy iz = "
                      << ix << " "
                      << iy << " "
                      << iz << std::endl;

            std::cout << "rx ry rz = "
                      << rx << " "
                      << ry << " "
                      << rz << std::endl;
        }
    }

    /**
     * Evaluate the split RT0 velocity on one subcell in local coordinates.
     *
     * Input:
     *   s          : subcell index, with
     *                  s = ix + 2*iy + 4*iz,  ix,iy,iz in {0,1}
     *   rx, ry, rz : local coordinates on that subcell, each in [0,1]
     *
     * Stored data:
     *   subcells[s].vn_outward[face]
     *
     * contains the outward normal velocity on each of the six faces of the
     * split subcell:
     *
     *   xm, xp, ym, yp, zm, zp
     *
     * Split RT0 interpolation:
     *   On an axis-aligned reference box [0,1]^3, the RT0 velocity field is
     *   component-wise affine and is uniquely determined by the six face-normal
     *   values. Using the outward-normal sign convention:
     *
     *     u_x(rx) = -vn[xm] * (1-rx) + vn[xp] * rx
     *     u_y(ry) = -vn[ym] * (1-ry) + vn[yp] * ry
     *     u_z(rz) = -vn[zm] * (1-rz) + vn[zp] * rz
     *
     * because:
     *   - on the negative face, the outward normal points in the negative axis
     *     direction, so vn[xm], vn[ym], vn[zm] enter with a minus sign;
     *   - on the positive face, the outward normal points in the positive axis
     *     direction, so vn[xp], vn[yp], vn[zp] enter with a plus sign.
     *
     * This returns the velocity in the split-cell reference-coordinate frame.
     */
    template<int dim>
    Tensor<1, dim> CellVelocityCacheRT0Split3D<dim>::interpolate_rt0_on_subcell(const unsigned int s,
        const double rx, const double ry, const double rz) const {
        AssertIndexRange(s, 8);
        AssertThrow(rx >= 0.0 && rx <= 1.0, dealii::ExcMessage("rx must be in [0,1]."));
        AssertThrow(ry >= 0.0 && ry <= 1.0, dealii::ExcMessage("ry must be in [0,1]."));
        AssertThrow(rz >= 0.0 && rz <= 1.0, dealii::ExcMessage("rz must be in [0,1]."));

        const auto &vn = subcells[s].vn_outward;

        Tensor<1, dim> u;

        // Because vn[xm] is outward on the -x face, it corresponds to the
        // negative x-direction. Likewise vn[xp] is outward on the +x face.
        // Therefore the affine x-component is:
        //
        //   ux(rx) = -vn[xm]*(1-rx) + vn[xp]*rx
        //
        // and similarly for y and z.
        u[0] = -vn[xm] * (1.0 - rx) + vn[xp] * rx;
        u[1] = -vn[ym] * (1.0 - ry) + vn[yp] * ry;
        u[2] = -vn[zm] * (1.0 - rz) + vn[zp] * rz;

        bool print_debug = false;
        if (print_debug)
        {
            std::cout << "subcell s = " << s << std::endl;
            std::cout << "vn[ym], vn[yp] = "
                      << vn[ym] << " "
                      << vn[yp] << std::endl;
            std::cout << "uy = " << u[1] << std::endl;
        }

        return u;
    }

    template <int dim>
    void CellVelocityCacheRT0Split3D<dim>::clear()
    {
        face_subface_vn = std::array<std::array<double, 4>, 6>();

        subcells = std::array<SubcellRT0Data, 8>();

        xv = std::array<double, 4>();
        yv = std::array<double, 4>();
        zb = std::array<double, 4>();
        zt = std::array<double, 4>();

        cell = CellIt();
    }

    template <int dim>
    bool CellVelocityCacheRT0Split3D<dim>::compute_reference_point(const Point<3> &x_phys,Point<3> &x_ref) const {
        double u, v;
        bool ok = getUV_Analytical(u, v, x_phys[0], x_phys[1], xv, yv);

        if (!ok || u < -1.2 || u > 1.2 || v < -1.2 || v > 1.2)
            ok = getUV_NewtonRaphson(u, v, x_phys[0], x_phys[1], xv, yv);

        if (!ok)
            return false;

        // Q1 shape functions on quad (u,v) in [-1,1]
        const double N0 = 0.25*(1-u)*(1-v);
        const double N1 = 0.25*(1+u)*(1-v);
        const double N2 = 0.25*(1+u)*(1+v);
        const double N3 = 0.25*(1-u)*(1+v);

        const double z_bottom = N0*zb[0] + N1*zb[1] + N2*zb[2] + N3*zb[3];
        const double z_top    = N0*zt[0] + N1*zt[1] + N2*zt[2] + N3*zt[3];

        const double H = z_top - z_bottom;
        if (H <= 0.0)
            return false;

        const double w = 2.0*(x_phys[2] - z_bottom)/H - 1.0; // [-1,1]

        x_ref[0] = u;
        x_ref[1] = v;
        x_ref[2] = w;

        // inside tolerance check (you can decide strictness)
        return (u >= -1.001 && u <= 1.001 &&
                v >= -1.001 && v <= 1.001 &&
                w >= -1.001 && w <= 1.001);

    }

    template <int dim>
    Point<dim> CellVelocityCacheRT0Split3D<dim>::map_reference_point_to_physical(const Point<dim> &x_ref, bool space01) const {
        static_assert(dim == 3, "map_reference_point_to_physical() is implemented for dim=3.");

        // Q1/trilinear map for an extruded hex defined by bottom quad (xv,yv,zb)
        // and top quad (xv,yv,zt). Canonical corner order:
        // bottom: 0=BL,1=BR,2=TR,3=TL and top: +4
        //
        // Cached arrays xv,yv,zb,zt are assumed to be in deal.II vertex indexing.
        // We reorder them through the same ring used for the wireframe.
        const int ring[4] = {0, 1, 3, 2};
        const auto lerp = [](const double a, const double b, const double t)
        {
            return a * (1.0 - t) + b * t;
        };

        // x_ref is assumed in [-1,1]^3
        double u = x_ref[0];
        double v = x_ref[1];
        double w = x_ref[2];
        if (space01)
        {
            u = 2.0 * x_ref[0] - 1.0;
            v = 2.0 * x_ref[1] - 1.0;
            w = 2.0 * x_ref[2] - 1.0;
        }

        // Q1 shape functions on the quad in (u,v)
        const double N0 = 0.25 * (1.0 - u) * (1.0 - v);
        const double N1 = 0.25 * (1.0 + u) * (1.0 - v);
        const double N2 = 0.25 * (1.0 + u) * (1.0 + v);
        const double N3 = 0.25 * (1.0 - u) * (1.0 + v);

        const int i0 = ring[0];
        const int i1 = ring[1];
        const int i2 = ring[2];
        const int i3 = ring[3];

        // bottom surface
        const double xb = N0 * xv[i0] + N1 * xv[i1] + N2 * xv[i2] + N3 * xv[i3];
        const double yb = N0 * yv[i0] + N1 * yv[i1] + N2 * yv[i2] + N3 * yv[i3];
        const double zb_ = N0 * zb[i0] + N1 * zb[i1] + N2 * zb[i2] + N3 * zb[i3];

        // top surface
        const double xt = N0 * xv[i0] + N1 * xv[i1] + N2 * xv[i2] + N3 * xv[i3];
        const double yt = N0 * yv[i0] + N1 * yv[i1] + N2 * yv[i2] + N3 * yv[i3];
        const double zt_ = N0 * zt[i0] + N1 * zt[i1] + N2 * zt[i2] + N3 * zt[i3];

        // linear blend along w, with w in [-1,1]
        const double t = 0.5 * (w + 1.0);

        Point<dim> x_phys;
        x_phys[0] = lerp(xb, xt, t);
        x_phys[1] = lerp(yb, yt, t);
        x_phys[2] = lerp(zb_, zt_, t);

        return x_phys;
    }



}

#endif //CACHED_VELOCITY_H
