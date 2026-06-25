//
// Created by giorgk on 6/25/26.
//

#ifndef NPSAT_FLOW_WRITE_OUTPUT_IMPL_H
#define NPSAT_FLOW_WRITE_OUTPUT_IMPL_H

template <int dim>
void NPSAT_FLOW<dim>::write_well_exchange_identity_csv_mpi(const std::string &prefix) const {
    if (!uo.print_well_resid_csv)
        return;

    // -----------------------------
    // 1) Local partial accumulation: well_id -> partial sums
    // -----------------------------
    struct Agg
    {
        std::uint32_t count_cells = 0;
        double sum_Qe      = 0.0;
        double sum_cwc     = 0.0;
        double sum_cwc_he  = 0.0;
    };

    std::unordered_map<std::uint32_t, Agg> local_acc;
    local_acc.reserve(1024);

    for (const auto &cell_entry : local_cell_well_map)
    {
        const auto &links = cell_entry.second;
        for (const auto &link : links)
        {
            const std::uint32_t w = static_cast<std::uint32_t>(link.well_global_index);
            const double cwc = (link.cwc_eff > 0.0) ? link.cwc_eff : link.cwc;

            auto &a = local_acc[w];
            a.sum_Qe     += link.Qe;
            a.sum_cwc    += cwc;
            a.sum_cwc_he += cwc * link.h_e;
            a.count_cells += 1;
        }
    }

    // -----------------------------
    // 2) Build per-destination send lists (route to well owner)
    // -----------------------------
    std::vector<std::vector<npsat_flow::WellIdentityPartial>> send_bins(n_proc);
    for (const auto &kv : local_acc)
    {
        const std::uint32_t w = kv.first;
        const Agg          &a = kv.second;

        const int owner = static_cast<int>(well_owner_rank[w]); // w % n_proc by your setup
        AssertThrow(owner >= 0 && owner < n_proc, dealii::ExcInternalError());

        npsat_flow::WellIdentityPartial p;
        p.well_id     = w;
        p.count_cells = a.count_cells;
        p.sum_Qe      = a.sum_Qe;
        p.sum_cwc     = a.sum_cwc;
        p.sum_cwc_he  = a.sum_cwc_he;

        send_bins[owner].push_back(p);
    }

    // -----------------------------
    // 3) MPI Alltoallv exchange of WellIdentityPartial (as raw bytes)
    // -----------------------------
    std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
    for (int r = 0; r < n_proc; ++r)
        send_counts[r] = static_cast<int>(send_bins[r].size());

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 mpi_communicator);

    std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
    int send_total = 0, recv_total = 0;
    for (int r = 0; r < n_proc; ++r)
    {
        send_displs[r] = send_total;
        send_total += send_counts[r];

        recv_displs[r] = recv_total;
        recv_total += recv_counts[r];
    }

    std::vector<npsat_flow::WellIdentityPartial> send_flat;
    send_flat.reserve(static_cast<std::size_t>(send_total));
    for (int r = 0; r < n_proc; ++r)
        for (const auto &p : send_bins[r])
            send_flat.push_back(p);

    std::vector<npsat_flow::WellIdentityPartial> recv_flat(static_cast<std::size_t>(recv_total));

    // counts/displs in BYTES for MPI_BYTE
    std::vector<int> send_counts_b(n_proc, 0), recv_counts_b(n_proc, 0);
    std::vector<int> send_displs_b(n_proc, 0), recv_displs_b(n_proc, 0);

    const int bytes_per = static_cast<int>(sizeof(npsat_flow::WellIdentityPartial));
    for (int r = 0; r < n_proc; ++r)
    {
        send_counts_b[r] = send_counts[r] * bytes_per;
        recv_counts_b[r] = recv_counts[r] * bytes_per;
        send_displs_b[r] = send_displs[r] * bytes_per;
        recv_displs_b[r] = recv_displs[r] * bytes_per;
    }

    MPI_Alltoallv(reinterpret_cast<const char *>(send_flat.data()), send_counts_b.data(), send_displs_b.data(), MPI_BYTE,
                reinterpret_cast<char *>(recv_flat.data()),       recv_counts_b.data(), recv_displs_b.data(), MPI_BYTE,
                mpi_communicator);

    // -----------------------------
    // 4) Owner ranks merge received partials into global sums for owned wells
    // -----------------------------
    std::unordered_map<std::uint32_t, Agg> owned_acc;
    owned_acc.reserve(static_cast<std::size_t>(recv_total));
    for (const auto &p : recv_flat)
    {
        auto &a = owned_acc[p.well_id];
        a.sum_Qe     += p.sum_Qe;
        a.sum_cwc    += p.sum_cwc;
        a.sum_cwc_he += p.sum_cwc_he;
        a.count_cells += p.count_cells;
    }

    // -----------------------------
    // 5) Write CSV for wells owned by this rank
    // -----------------------------
    const std::string step = Utilities::int_to_string(time_tracking.simulation_step(), 3);
    const std::string str_rank = Utilities::int_to_string(my_rank, 4);
    std::ostringstream fname;
    fname << prefix << "_well_resid_rank_" << str_rank << "_step_" << step <<  ".csv";

    std::vector<std::string> local_lines;
    local_lines.reserve(owned_acc.size());

    std::ofstream out(fname.str());
    AssertThrow(out.good(), dealii::ExcMessage("Could not open output file: " + fname.str()));

    out << "well_id,n_cells,sum_Qe,sum_cwc,sum_cwc_he,hw,sum_Qe_from_identity,residual\n";
    out << std::setprecision(16) << std::scientific;

    const unsigned int n_wells = mnwells.wells.size();

    for (unsigned int w = 0; w < n_wells; ++w)
    {
        if (static_cast<int>(well_owner_rank[w]) != my_rank)
            continue;

        const auto it = owned_acc.find(static_cast<std::uint32_t>(w));
        const Agg a = (it != owned_acc.end() ? it->second : Agg{});

        // Owner has the well head (owned distribution). If your well_solution is ghosted,
        // this is still fine; if it's owned-only, we are accessing only owned indices here.
        const double hw = well_solution[w];

        const double sum_Qe_from_identity = a.sum_cwc * hw - a.sum_cwc_he;
        const double residual            = a.sum_Qe - sum_Qe_from_identity;

        out << w << ","
        << a.count_cells << ","
        << a.sum_Qe << ","
        << a.sum_cwc << ","
        << a.sum_cwc_he << ","
        << hw << ","
        << sum_Qe_from_identity << ","
        << residual << "\n";
    }
    out.flush();
}

template <int dim>
void NPSAT_FLOW<dim>::compute_wellbore_flows(const std::string &prefix) const {
    if (!uo.print_wellboreflow_csv)
        return;

    pcout << "Computing wellbore flows (MNW)..." << std::endl;
    const unsigned int n_wells = mnwells.wells.size();

    // ---------------------------------------------------------------------
    // Conventions used here (match the current working setup):
    //
    //   Qe = CWC_e * (h_w - H_e)
    //
    // We observed:
    //   Qe < 0  => pumping (aquifer -> well)   [adds water into well]
    //   Qe > 0  => injection (well -> aquifer) [removes water from well]
    //
    // Define wellbore flow F as POSITIVE UPWARD.
    //
    // For a node/cell segment, conservation gives:
    //   F_top = F_bottom - Qe
    //
    // Because if Qe is negative (aquifer -> well), then -Qe is positive inflow
    // and upward flow increases as we move upward.
    //
    // Bottom boundary is closed:
    //   F_bottom(bottommost) = 0
    //
    // Then the computed F_top at the topmost node should be approximately
    //   F_top(top) = -sum(Qe over nodes)  ≈  -Qtot
    // (with Qtot negative for pumping wells).
    // ---------------------------------------------------------------------

    // ---------------------------------------------------------------------
    // 1) Build local segment list from local_cell_well_map
    //    (Qe and ze must already be computed, e.g. by compute_fluxes()).
    // ---------------------------------------------------------------------
    std::vector<npsat_flow::WellSegmentMsg> local_segments;
    local_segments.reserve(1024);

    for (const auto &cell_pair : local_cell_well_map)
    {
        const auto &links = cell_pair.second;
        for (const auto &link : links)
        {
            const unsigned int w = mnwells.wells[link.well_global_index].global_index;
            if (w >= n_wells)
                continue;

            npsat_flow::WellSegmentMsg s;
            s.well_id = static_cast<std::uint32_t>(w);
            s.ze      = link.ze;
            s.Qe      = link.Qe; // must already be computed
            local_segments.push_back(s);
        }
    }

    // ---------------------------------------------------------------------
    // 2) Route segments to well owner ranks
    // ---------------------------------------------------------------------
    std::vector<std::vector<npsat_flow::WellSegmentMsg>> send_bins(n_proc);
    for (const auto &s : local_segments)
    {
        const unsigned int w = static_cast<unsigned int>(s.well_id);
        AssertThrow(w < well_owner_rank.size(), dealii::ExcInternalError());
        const int owner = static_cast<int>(well_owner_rank[w]);
        AssertThrow(owner >= 0 && owner < n_proc, dealii::ExcInternalError());
        send_bins[owner].push_back(s);
    }

    std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
    for (int r = 0; r < n_proc; ++r)
        send_counts[r] = static_cast<int>(send_bins[r].size());

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
                 recv_counts.data(), 1, MPI_INT,
                 mpi_communicator);

    std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
    int send_total = 0, recv_total = 0;
    for (int r = 0; r < n_proc; ++r)
    {
        send_displs[r] = send_total;
        send_total += send_counts[r];

        recv_displs[r] = recv_total;
        recv_total += recv_counts[r];
    }

    std::vector<npsat_flow::WellSegmentMsg> send_flat;
    send_flat.reserve(static_cast<std::size_t>(send_total));
    for (int r = 0; r < n_proc; ++r)
        for (const auto &s : send_bins[r])
            send_flat.push_back(s);

    std::vector<npsat_flow::WellSegmentMsg> recv_flat(static_cast<std::size_t>(recv_total));

    const int bytes_per = static_cast<int>(sizeof(npsat_flow::WellSegmentMsg));
    std::vector<int> send_counts_b(n_proc, 0), recv_counts_b(n_proc, 0);
    std::vector<int> send_displs_b(n_proc, 0), recv_displs_b(n_proc, 0);
    for (int r = 0; r < n_proc; ++r)
    {
        send_counts_b[r] = send_counts[r] * bytes_per;
        recv_counts_b[r] = recv_counts[r] * bytes_per;
        send_displs_b[r] = send_displs[r] * bytes_per;
        recv_displs_b[r] = recv_displs[r] * bytes_per;
    }

    MPI_Alltoallv(reinterpret_cast<const char *>(send_flat.data()), send_counts_b.data(), send_displs_b.data(), MPI_BYTE,
                reinterpret_cast<char *>(recv_flat.data()),       recv_counts_b.data(), recv_displs_b.data(), MPI_BYTE,
                mpi_communicator);

    // ---------------------------------------------------------------------
    // 3) On each rank, group received segments by well_id (these are owned wells)
    // ---------------------------------------------------------------------
    struct Seg { double ze; double Qe; };

    std::unordered_map<std::uint32_t, std::vector<Seg>> per_well;
    per_well.reserve(1024);

    for (const auto &s : recv_flat)
        per_well[s.well_id].push_back(Seg{s.ze, s.Qe});

    // ---------------------------------------------------------------------
    // 4) Write one CSV per rank with owned wells only.
    //    Variable number of columns is allowed as requested.
    // ---------------------------------------------------------------------
    const std::string step = Utilities::int_to_string(time_tracking.simulation_step(), 3);
    const std::string str_rank = Utilities::int_to_string(my_rank, 4);

    std::ostringstream fname;
    fname << prefix << "_wellboreflow_rank_" << str_rank << "_step_" << step << ".csv";

    std::ofstream out(fname.str());
    AssertThrow(out.good(), dealii::ExcMessage("Could not open output file: " + fname.str()));

    // Header: fixed part + repeating triplets (ze_k, Qe_k, Qwbf_k)
    // Column count varies by well; that is acceptable per your request.
    out << "well_id,Eid,Qtot,n_segments,F_top,-Qtot,diff";
    out << ",(ze_0,Qe_0,Qwbf_0)...\n";
    out << std::setprecision(16) << std::scientific;

    for (unsigned int w = 0; w < n_wells; ++w)
    {
        if (static_cast<int>(well_owner_rank[w]) != my_rank)
            continue;

        const double Qtot = mnwells.pumping_rate(mnwells.wells[w].q_row);

        auto it = per_well.find(static_cast<std::uint32_t>(w));
        if (it == per_well.end() || it->second.empty())
        {
            // Owned well with no segments on any rank (unlikely, but handle it)
            out << w << "," << mnwells.wells[w].Eid << "," << Qtot
                << "," << 0 << "," << 0.0 << "," << (-Qtot) << "," << (0.0 + Qtot) << "\n";
            continue;
        }

        auto &segs = it->second;

        // Sort bottom -> top by ze (tested implementation logic)
        std::sort(segs.begin(), segs.end(),
                  [](const Seg &a, const Seg &b) { return a.ze < b.ze; });

        // Accumulate wellbore flows using the tested implementation:
        // F_bottom starts at 0, and F_top = F_bottom - Qe
        double F_bottom = 0.0;

        // We will output per segment: ze, Qe, Qwbf (=F_top at that segment)
        // and diagnostics at the end.
        std::vector<double> qwbfs;
        qwbfs.reserve(segs.size());

        for (const auto &seg : segs)
        {
            const double F_top = F_bottom - seg.Qe;
            qwbfs.push_back(F_top);
            F_bottom = F_top;
        }

        const double F_top_well = F_bottom;
        //const double Qtot       = Qtot;
        const double diff       = F_top_well + Qtot;

        // Fixed columns
        out << w << ","
            << mnwells.wells[w].Eid << ","
            << Qtot << ","
            << static_cast<unsigned int>(segs.size()) << ","
            << F_top_well << ","
            << (-Qtot) << ","
            << diff;

        // Variable columns: (ze_k, Qe_k, Qwbf_k)
        for (std::size_t k = 0; k < segs.size(); ++k)
        {
            out << ","
                << segs[k].ze << ","
                << segs[k].Qe << ","
                << qwbfs[k];
        }
        out << "\n";
    }
    out.flush();
    pcout << "Wellbore flow computation complete." << std::endl;
}

template<int dim>
void NPSAT_FLOW<dim>::write_wellbore_segments_csv_mpi(const std::string &prefix) const {

    const bool write_csv = uo.print_wellbore_segments_csv;
    const bool write_vtu = uo.print_wellbore_segments_vtu;
    const bool write_legacy_vtk = uo.print_wellbore_segments_legacy_vtk;
    const bool write_trace_binary = uo.save_trace_data;

    if (!write_csv && !write_vtu && !write_legacy_vtk && !write_trace_binary)
        return;

    const unsigned int n_wells = mnwells.wells.size();

    // -----------------------------
    // 1) Local segments (now take z_bot/z_top directly from CellWellLink)
    // -----------------------------
    struct WellSegMsg2Z
    {
        std::uint32_t well_global_index; // internal global well index (not Eid)
        std::uint32_t cell_id;
        std::uint16_t cell_owner_rank;   // NEW: rank that owns the cell
        std::uint16_t pad0;

        double        ze;
        double        Qe;
        double        cwc;
        double        z_bot;
        double        z_top;
    };
#pragma pack(push, 1)
    struct WellSegFlowMsg
    {
        std::uint32_t well_global_index;
        std::uint32_t cell_id;      // active_cell_index
        double        ze;           // optional but useful for debugging
        double        Qe;
        double        Qwbf_bot;     // flow crossing bottom of the segment (positive upward)
        double        Qwbf_top;     // flow crossing top of the segment (positive upward)
    };
#pragma pack(pop)
    static_assert(std::is_trivially_copyable<WellSegMsg2Z>::value, "WellSegMsg2Z must be trivially copyable.");
    static_assert(std::is_trivially_copyable<WellSegFlowMsg>::value, "WellSegFlowMsg must be trivially copyable.");

    std::vector<WellSegMsg2Z> local;
    local.reserve(1024);

}

#endif //NPSAT_FLOW_WRITE_OUTPUT_IMPL_H
