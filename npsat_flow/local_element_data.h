//
// Created by giorgk on 6/24/26.
//

#ifndef LOCAL_ELEMENT_DATA_H
#define LOCAL_ELEMENT_DATA_H

#include <deal.II/base/exceptions.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>


namespace npsat_flow{
    using namespace dealii;

    enum class Layer : std::uint8_t
    {
        nonlinear  = 1,
        postprocess = 2
    };

    // ---------------------------
    // Fixed-size copy helpers
    // ---------------------------
    template <int m, int n>
    void array_to_fullmatrix(const std::array<double, m * n> &src,
                             dealii::FullMatrix<double>           &dst)
    {
        AssertDimension(dst.m(), m);
        AssertDimension(dst.n(), n);

        // Row-major layout: src[i*n + j]
        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                dst(i, j) = src[static_cast<std::size_t>(i * n + j)];
    }

    template <int m, int n>
    void fullmatrix_to_array(const dealii::FullMatrix<double> &src,
                             std::array<double, m * n>       &dst)
    {
        AssertDimension(src.m(), m);
        AssertDimension(src.n(), n);

        for (int i = 0; i < m; ++i)
            for (int j = 0; j < n; ++j)
                dst[static_cast<std::size_t>(i * n + j)] = src(i, j);
    }

    template <int n>
    void array_to_vector(const std::array<double, n> &src,
                         dealii::Vector<double>     &dst)
    {
        AssertDimension(dst.size(), n);
        for (int i = 0; i < n; ++i)
            dst[static_cast<unsigned int>(i)] = src[static_cast<std::size_t>(i)];
    }

    template <int n>
    void vector_to_array(const dealii::Vector<double> &src,
                         std::array<double, n>        &dst)
    {
        AssertDimension(src.size(), n);
        for (int i = 0; i < n; ++i)
            dst[static_cast<std::size_t>(i)] = src[static_cast<unsigned int>(i)];
    }


    /**
     * LocalElementDataRT0DG0
     *
     * Two-layer cache:
     *  - Layer 1 (always allocated): data needed during nonlinear iterations
     *      Shat (6x6), rhs (6), Kinv (1), E (1x6)
     *  - Layer 2 (allocated on demand): postprocess / recovery auxiliaries
     *      A_inv (6x6), B (1x6), C (6x6)
     *
     * Storage is contiguous and allocation-free per cell (no per-cell heap).
     * Slot indexing is assumed to be a compact local index [0, n_local_cells).
     */
    class LocalElementDataRT0DG0
    {
    public:
        static constexpr int n_trace = 6; // RT0 on hex faces (trace space here assumed 6 dofs)
        static constexpr int n_flux  = 6; // A_inv is 6x6 in your formulation
        static constexpr int n_head  = 1; // DG0

        using Mat6  = std::array<double, 36>; // 6x6 row-major
        using Vec6  = std::array<double, 6>;
        using Row16 = std::array<double, 6>;  // 1x6 stored as length-6
        using Sca1  = double;

        LocalElementDataRT0DG0() = default;
        unsigned int size() const noexcept { return n_cells_; }

        void initialize(const unsigned int n_local_cells) {
            n_cells_ = n_local_cells;

            e_.assign(n_cells_, Row16{});
            kinv_.assign(n_cells_, 0.0);
            v_.assign(n_cells_, 0.0);
            a_inv_.assign(n_cells_, Mat6{});
            b_.assign(n_cells_, Row16{});
            c_.assign(n_cells_, Mat6{});
            kinv_v_.assign(n_cells_, 0.0);
            m_.assign(n_cells_, 0.0);
            ae_.assign(n_cells_, Vec6{});
            flags_.assign(n_cells_, 0u);
        }

        // -------------
        // Slot checks
        // -------------
        void assert_valid_slot(const unsigned int slot) const
        {
            AssertIndexRange(slot, n_cells_);
        }

        // -------------
        // Flags
        // -------------
        std::uint8_t get_flag(const unsigned int slot) const
        {
            assert_valid_slot(slot);
            return flags_[slot];
        }

        void set_flag(const unsigned int slot, const std::uint8_t value)
        {
            assert_valid_slot(slot);
            flags_[slot] = value;
        }

        double get_kinv(const unsigned int slot) const
        {
            assert_valid_slot(slot);
            return kinv_[slot];
        }
        void   set_kinv(const unsigned int slot, const double v)
        {
            assert_valid_slot(slot);
            kinv_[slot] = v;
        }

        double get_kinv_v(const unsigned int slot) const
        {
            assert_valid_slot(slot);
            return kinv_v_[slot];
        }
        void   set_kinv_v(const unsigned int slot, const double v)
        {
            assert_valid_slot(slot);
            kinv_v_[slot] = v;
        }

        double get_M00(const unsigned int slot) const
        {
            assert_valid_slot(slot);
            return m_[slot];
        }
        void set_M00(const unsigned int slot, const double v)
        {
            assert_valid_slot(slot);
            m_[slot] = v;
        }

        double get_V(const unsigned int slot) const
        {
            assert_valid_slot(slot);
            AssertThrow(!v_.empty(), dealii::ExcMessage("Nonlinear layer not allocated (V)."));
            return v_[slot];
        }

        void set_V(const unsigned int slot, const double val)
        {
            assert_valid_slot(slot);
            AssertThrow(!v_.empty(), dealii::ExcMessage("Nonlinear layer not allocated (V)."));
            v_[slot] = val;
        }

        const Row16 &E_array(const unsigned int slot) const { assert_valid_slot(slot); return e_[slot]; }
        Row16       &E_array(const unsigned int slot)       { assert_valid_slot(slot); return e_[slot]; }

        void get_ae(const unsigned int slot, dealii::FullMatrix<double> &dst) const
        {
            assert_valid_slot(slot);
            array_to_fullmatrix<1, 6>(ae_[slot], dst);
        }

        void set_ae(const unsigned int slot, const dealii::FullMatrix<double> &src)
        {
            assert_valid_slot(slot);
            fullmatrix_to_array<1, 6>(src, ae_[slot]);
        }

        // ---- E (1x6 stored as length-6)
        void get_E(const unsigned int slot, dealii::FullMatrix<double> &dst) const
        {
            assert_valid_slot(slot);
            AssertDimension(dst.m(), 1);
            AssertDimension(dst.n(), 6);

            for (int j = 0; j < 6; ++j)
                dst(0, j) = e_[slot][static_cast<std::size_t>(j)];
        }

        void set_E(const unsigned int slot, const dealii::FullMatrix<double> &src)
        {
            assert_valid_slot(slot);
            AssertDimension(src.m(), 1);
            AssertDimension(src.n(), 6);

            for (int j = 0; j < 6; ++j)
                e_[slot][static_cast<std::size_t>(j)] = src(0, j);
        }

        // Overload: E as Vector(6) if that is more convenient
        void get_E(const unsigned int slot, dealii::Vector<double> &dst) const
        {
            assert_valid_slot(slot);
            array_to_vector<6>(e_[slot], dst);
        }

        void set_E(const unsigned int slot, const dealii::Vector<double> &src)
        {
            assert_valid_slot(slot);
            vector_to_array<6>(src, e_[slot]);
        }

        // ---- A_inv (6x6)
        void get_A_inv(const unsigned int slot, dealii::FullMatrix<double> &dst) const
        {
            assert_valid_slot(slot);
            //AssertThrow(a_inv_, dealii::ExcMessage("A_inv not allocated."));
            array_to_fullmatrix<6, 6>(a_inv_[slot], dst);
        }

        void set_A_inv(const unsigned int slot, const dealii::FullMatrix<double> &src)
        {
            assert_valid_slot(slot);
            //AssertThrow(a_inv_, dealii::ExcMessage("A_inv not allocated."));
            fullmatrix_to_array<6, 6>(src, a_inv_[slot]);
        }

        // ---- B (1x6)
        void get_B(const unsigned int slot, dealii::FullMatrix<double> &dst) const
        {
            assert_valid_slot(slot);
            //AssertThrow(b_, dealii::ExcMessage("B not allocated."));
            AssertDimension(dst.m(), 1);
            AssertDimension(dst.n(), 6);

            //const auto &arr = (*b_)[slot];
            for (int j = 0; j < 6; ++j)
                dst(0, j) = b_[slot][static_cast<std::size_t>(j)];
        }

        void set_B(const unsigned int slot, const dealii::FullMatrix<double> &src)
        {
            assert_valid_slot(slot);
            //AssertThrow(b_, dealii::ExcMessage("B not allocated."));
            AssertDimension(src.m(), 1);
            AssertDimension(src.n(), 6);

            //auto &arr = (*b_)[slot];
            for (int j = 0; j < 6; ++j)
                b_[slot][static_cast<std::size_t>(j)] = src(0, j);
        }

        // ---- C (6x6)
        void get_C(const unsigned int slot, dealii::FullMatrix<double> &dst) const
        {
            assert_valid_slot(slot);
            //AssertThrow(c_, dealii::ExcMessage("C not allocated."));
            array_to_fullmatrix<6, 6>(c_[slot], dst);
        }

        void set_C(const unsigned int slot, const dealii::FullMatrix<double> &src)
        {
            assert_valid_slot(slot);
            //AssertThrow(c_, dealii::ExcMessage("C not allocated."));
            fullmatrix_to_array<6, 6>(src, c_[slot]);
        }

        Layer get_assembly_layer() const
        {
            return assembly_layer_;
        }

        void set_assembly_layer(const Layer layer)
        {
            assembly_layer_ = layer;
        }
    private:
        unsigned int n_cells_ = 0;
        std::vector<Mat6> a_inv_;
        std::vector<Mat6>  c_;
        std::vector<Vec6>  ae_;    // 6 per cell
        std::vector<double> kinv_;  // scalar per cell
        std::vector<double> kinv_v_;  // scalar per cell
        std::vector<Row16> e_;      // 1x6 per cell (stored as length-6)
        std::vector<Row16> b_;
        std::vector<std::uint8_t> flags_;
        std::vector<double> v_; // scalar V = M*h_old + dt*F per cell (DG0)
        std::vector<double> m_;
        Layer assembly_layer_ = Layer::nonlinear;
    };
}

#endif //LOCAL_ELEMENT_DATA_H
