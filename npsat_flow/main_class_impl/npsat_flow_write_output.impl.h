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

    for (const auto &cell_pair : local_cell_well_map) {
        const unsigned int cell_id = cell_pair.first; // already store active_cell_index() as key
        const auto        &links   = cell_pair.second;
        for (const auto &link : links)
        {
            const unsigned int w = mnwells.wells[link.well_global_index].global_index;
            if (w >= n_wells) continue;

            WellSegMsg2Z s;
            s.well_global_index = static_cast<std::uint32_t>(w);
            s.cell_id           = static_cast<std::uint32_t>(cell_id);
            s.cell_owner_rank   = static_cast<std::uint16_t>(my_rank);
            s.pad0    = 0;
            s.ze      = link.ze;
            s.Qe      = link.Qe;
            s.cwc     = (link.cwc_eff > 0.0) ? link.cwc_eff : link.cwc;
            s.z_bot = link.w_zbot;
            s.z_top = link.w_zbot + link.sl;
            local.push_back(s);
        }
    }

    // -----------------------------
    // 2) Route to owner (by global well index)
    // -----------------------------
    std::vector<std::vector<WellSegMsg2Z>> send_bins(n_proc);
    for (const auto &s : local)
    {
        const unsigned int w = static_cast<unsigned int>(s.well_global_index);
        const int owner = static_cast<int>(well_owner_rank[w]);
        AssertThrow(owner >= 0 && owner < n_proc, ExcInternalError());
        send_bins[owner].push_back(s);
    }

    std::vector<int> send_counts(n_proc, 0), recv_counts(n_proc, 0);
    for (int r = 0; r < n_proc; ++r)
        send_counts[r] = static_cast<int>(send_bins[r].size());

    MPI_Alltoall(send_counts.data(), 1, MPI_INT,
               recv_counts.data(), 1, MPI_INT,
               mpi_communicator);

    std::vector<int> send_displs(n_proc, 0), recv_displs(n_proc, 0);
    int send_total=0, recv_total=0;
    for (int r = 0; r < n_proc; ++r)
    {
        send_displs[r] = send_total; send_total += send_counts[r];
        recv_displs[r] = recv_total; recv_total += recv_counts[r];
    }

    std::vector<WellSegMsg2Z> send_flat;
    send_flat.reserve(send_total);
    for (int r = 0; r < n_proc; ++r)
        for (const auto &s : send_bins[r])
            send_flat.push_back(s);

    std::vector<WellSegMsg2Z> recv_flat(recv_total);

    const int bytes_per = static_cast<int>(sizeof(WellSegMsg2Z));
    std::vector<int> send_counts_b(n_proc), recv_counts_b(n_proc),
                   send_displs_b(n_proc), recv_displs_b(n_proc);

    for (int r = 0; r < n_proc; ++r)
    {
        send_counts_b[r] = send_counts[r] * bytes_per;
        recv_counts_b[r] = recv_counts[r] * bytes_per;
        send_displs_b[r] = send_displs[r] * bytes_per;
        recv_displs_b[r] = recv_displs[r] * bytes_per;
    }

    MPI_Alltoallv(reinterpret_cast<const char *>(send_flat.data()),
                send_counts_b.data(), send_displs_b.data(), MPI_BYTE,
                reinterpret_cast<char *>(recv_flat.data()),
                recv_counts_b.data(), recv_displs_b.data(), MPI_BYTE,
                mpi_communicator);

    // -----------------------------
    // 3) Group by global well index on owner
    // -----------------------------
    struct Seg
    {
        std::uint32_t cell_id;
        std::uint16_t cell_owner_rank; // NEW
        std::uint16_t pad0;
        double ze;
        double Qe;
        double cwc;
        double z_bot;
        double z_top;
    };

    std::unordered_map<std::uint32_t, std::vector<Seg>> by_well;
    by_well.reserve(1024);
    for (const auto &s : recv_flat)
    {
        by_well[s.well_global_index].push_back(Seg{s.cell_id, s.cell_owner_rank, 0,
                                                s.ze, s.Qe, s.cwc, s.z_bot, s.z_top});
    }

    // -----------------------------
    // 4) Write CSV + accumulate payload for VTK writers
    //    IMPORTANT: VTK "well id" must be named "Eid" and taken from MNWell::Eid.
    // -----------------------------
    std::vector<npsat_flow::WellSegmentOut> vtk_segments;
    vtk_segments.reserve(recv_flat.size());

    const std::string step = Utilities::int_to_string(time_tracking.simulation_step(), 3);
    const std::string str_rank = Utilities::int_to_string(my_rank, 4);

    std::ostringstream csvname;
    csvname << prefix << "_wellbore_segments_rank_" << str_rank << "_step_" << step << ".csv";
    std::ofstream out;
    if (write_csv)
    {
        out.open(csvname.str());
        AssertThrow(out.good(), ExcMessage("Could not open: " + csvname.str()));

        out << "well_id,Eid,cell_id,ze,Qe,cwc,Qwbf_top,Qtot,F_top_well,diff\n";
        out << std::setprecision(3) << std::fixed;
    }
    std::vector<std::vector<WellSegFlowMsg>> send_back(n_proc);

    for (unsigned int w=0; w<n_wells; ++w)
    {
        if (static_cast<int>(well_owner_rank[w]) != my_rank)
            continue;

        auto it = by_well.find(static_cast<std::uint32_t>(w));
        if (it == by_well.end() || it->second.empty())
            continue;

        auto &segs = it->second;
        std::sort(segs.begin(), segs.end(),
                  [](const Seg &a, const Seg &b){ return a.ze < b.ze; });

        // Qwbf_top logic per segment
        double F_bottom = 0.0;
        std::vector<double> F_bot(segs.size());
        std::vector<double> F_top(segs.size());
        for (std::size_t k=0;k<segs.size();++k){
            F_bot[k] = F_bottom;
            F_top[k] = F_bottom - segs[k].Qe; // tested logic
            F_bottom = F_top[k];
        }

        const double F_top_well = F_bottom;
        const double Qtot = mnwells.pumping_rate(mnwells.wells[w].q_row);
        const double diff = F_top_well + Qtot;

        const int    Eid = mnwells.wells[w].Eid; // REQUIRED identifier for VTK
        const double x   = mnwells.wells[w].x;
        const double y   = mnwells.wells[w].y;

        for (std::size_t k=0;k<segs.size();++k)
        {
            if (write_csv)
            {
                out << w << ","
                    << Eid << ","
                    << segs[k].cell_id << ","
                    << segs[k].ze << ","
                    << segs[k].Qe << ","
                    << segs[k].cwc << ","
                    << F_top[k] << ","
                    << Qtot << ","
                    << F_top_well << ","
                    << diff << "\n";
            }

            // Payload for VTK writers
            npsat_flow::WellSegmentOut s;
            s.x = x;
            s.y = y;
            s.z_bot = segs[k].z_bot;
            s.z_top = segs[k].z_top;

            s.Eid = Eid;
            s.cell_id = segs[k].cell_id;
            s.ze = segs[k].ze;

            s.Qe = segs[k].Qe;
            s.cwc = segs[k].cwc;
            s.Qwbf_top = F_top[k];
            s.Qtot = Qtot;
            s.F_top_well = F_top_well;
            s.diff = diff;

            vtk_segments.push_back(s);

            WellSegFlowMsg m;
            m.well_global_index = static_cast<std::uint32_t>(w);
            m.cell_id  = segs[k].cell_id;
            m.ze       = segs[k].ze;
            m.Qe       = segs[k].Qe;
            m.Qwbf_bot = F_bot[k];
            m.Qwbf_top = F_top[k];

            const int owner = static_cast<int>(segs[k].cell_owner_rank);
            AssertThrow(owner >= 0 && owner < n_proc, ExcInternalError());
            send_back[owner].push_back(m);
        }
    }
    if (write_csv)
        out.flush();

    // ---- back exchange counts
    std::vector<int> send_counts2(n_proc, 0), recv_counts2(n_proc, 0);
    for (int r=0; r<n_proc; ++r)
        send_counts2[r] = static_cast<int>(send_back[r].size());

    MPI_Alltoall(send_counts2.data(), 1, MPI_INT,
                 recv_counts2.data(), 1, MPI_INT,
                 mpi_communicator);

    std::vector<int> send_displs2(n_proc, 0), recv_displs2(n_proc, 0);
    int send_total2 = 0, recv_total2 = 0;
    for (int r=0; r<n_proc; ++r)
    {
        send_displs2[r] = send_total2; send_total2 += send_counts2[r];
        recv_displs2[r] = recv_total2; recv_total2 += recv_counts2[r];
    }

    // ---- flatten
    std::vector<WellSegFlowMsg> send_flat2;
    send_flat2.reserve(send_total2);
    for (int r=0; r<n_proc; ++r)
        for (const auto &m : send_back[r])
            send_flat2.push_back(m);

    std::vector<WellSegFlowMsg> recv_flat2(recv_total2);

    // ---- bytes
    const int bytes_per2 = static_cast<int>(sizeof(WellSegFlowMsg));
    std::vector<int> send_counts2_b(n_proc), recv_counts2_b(n_proc),
                     send_displs2_b(n_proc), recv_displs2_b(n_proc);
    for (int r=0; r<n_proc; ++r)
    {
        send_counts2_b[r] = send_counts2[r] * bytes_per2;
        recv_counts2_b[r] = recv_counts2[r] * bytes_per2;
        send_displs2_b[r] = send_displs2[r] * bytes_per2;
        recv_displs2_b[r] = recv_displs2[r] * bytes_per2;
    }

    MPI_Alltoallv(reinterpret_cast<const char *>(send_flat2.data()),
                  send_counts2_b.data(), send_displs2_b.data(), MPI_BYTE,
                  reinterpret_cast<char *>(recv_flat2.data()),
                  recv_counts2_b.data(), recv_displs2_b.data(), MPI_BYTE,
                  mpi_communicator);

    // -----------------------------
    // 5) Output VTK using the two separated writers
    // -----------------------------
    if (write_vtu)
    {
        std::ostringstream vtu;
        vtu << prefix << "_wellbore_segments_rank_" << str_rank << "_step_" << step << ".vtu";
        npsat_flow::write_wells_as_dataout_1d3(vtu.str(), vtk_segments);
    }
    if (write_legacy_vtk)
    {
        std::ostringstream vtk;
        vtk << prefix << "_wellbore_segments_rank_" << str_rank << "_step_" << step << "_legacy.vtk";
        npsat_flow::write_wells_as_legacy_vtk_polydata(vtk.str(), vtk_segments);
    }
    if (write_trace_binary)
    {
        const std::string step = Utilities::int_to_string(time_tracking.simulation_step(), 3);
        const std::string str_rank = Utilities::int_to_string(my_rank, 4);

        std::ostringstream binname;
        binname << prefix << "_particle_well_flows_rank_" << str_rank << "_step_" << step << ".bin";
        std::ofstream bout(binname.str(), std::ios::binary);
        AssertThrow(bout.good(), ExcMessage("Could not open: " + binname.str()));

        const char magic[8] = {'P','W','F','L','O','W','v','1'};
        bout.write(magic, 8);

        const std::uint32_t version = 1;
        const std::uint32_t step_u32 = static_cast<std::uint32_t>(time_tracking.simulation_step());
        npsat_flow::write_pod(bout, version);
        npsat_flow::write_pod(bout, step_u32);

        const std::uint64_t nrec = static_cast<std::uint64_t>(recv_flat2.size());
        npsat_flow::write_pod(bout, nrec);

        // Write records
        for (const auto &m : recv_flat2)
        {
            npsat_flow::write_pod(bout, m.cell_id);
            npsat_flow::write_pod(bout, m.well_global_index);
            npsat_flow::write_pod(bout, m.Qe);
            npsat_flow::write_pod(bout, m.Qwbf_bot);
            npsat_flow::write_pod(bout, m.Qwbf_top);
        }

        bout.flush();
    }
}

template <int dim>
void NPSAT_FLOW<dim>::output_results(const std::string &prefix) {

    const bool write_csv = uo.print_cell_budget_csv;
    const bool write_solution_cell_vtu = uo.print_solution_cell_vtu;
    const bool write_cellcenters_vtk = uo.print_cellcenters_vtk;
    const bool write_q_to_vtu = uo.print_q_to_vtu && (write_solution_cell_vtu || write_cellcenters_vtk);
    const bool write_lambda_to_vtu = uo.print_lambda_to_vtu && (write_solution_cell_vtu || write_cellcenters_vtk);
    const bool write_area_to_vtu = uo.print_area_to_vtu && (write_solution_cell_vtu || write_cellcenters_vtk);

    if (!write_csv && !write_solution_cell_vtu && !write_cellcenters_vtk)
        return;

    pcout << "Writing CELL-based results to VTU/CSV at output step "
          << time_tracking.simulation_step() << "..." << std::endl;

    // -----------------------------
    // Output file names
    // -----------------------------
    const std::string step = Utilities::int_to_string(time_tracking.simulation_step(), 3);
    const std::string str_rank = Utilities::int_to_string(my_rank, 4);
    std::ostringstream csv_name;
    csv_name << prefix << "_cell_budget_rank_" << str_rank << "_step_" << step << ".csv";

    std::ofstream csv;
    if (write_csv)
    {
        csv.open(csv_name.str());
        AssertThrow(csv.good(), ExcMessage("Could not open CSV: " + csv_name.str()));

        csv << "cell_id,Xc,Yc,Zc,"
            << "dof0,dof1,dof2,dof3,dof4,dof5,"
            << "Qface0,Qface1,Qface2,Qface3,Qface4,Qface5,"
            << "Qw,Qwbf,dS_rate,head,"
            << "lambda0,lambda1,lambda2,lambda3,lambda4,lambda5\n";
    }

    // -----------------------------
    // DG0 vectors (cell data) sized to the DG0 DoFHandler
    // These are distributed and scale in MPI.
    // -----------------------------
    // Reuse the same partitioning as h_new/h_old (DG0 head space).
    // If you have head_locally_owned_dofs/head_locally_relevant_dofs members,
    // use those. Otherwise, use h_new.locally_owned_elements() etc.
    TrilinosWrappers::MPI::Vector head_out;
    TrilinosWrappers::MPI::Vector dS_out;
    TrilinosWrappers::MPI::Vector Qw_out;

    head_out.reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);
    dS_out.reinit  (head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);
    Qw_out.reinit  (head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);

    // Optional per-face DG0 vectors (each face quantity stored as one DG0 field)
    std::array<TrilinosWrappers::MPI::Vector, GeometryInfo<dim>::faces_per_cell> qface_out;
    std::array<TrilinosWrappers::MPI::Vector, GeometryInfo<dim>::faces_per_cell> lambda_out;
    std::array<TrilinosWrappers::MPI::Vector, GeometryInfo<dim>::faces_per_cell> area_out;

    if (write_q_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            qface_out[f].reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);

    if (write_lambda_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            lambda_out[f].reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);

    if (write_area_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            area_out[f].reinit(head_locally_owned_dofs, head_locally_relevant_dofs, mpi_communicator);

    // -----------------------------
    // Face quadrature for flux integration
    // -----------------------------
    QGauss<dim-1> face_quad(fe_flux.degree + 2);
    FEFaceValues<dim> fe_face(fe_flux, face_quad,
                              update_values | update_normal_vectors | update_JxW_values);
    FEValuesExtractors::Vector flux(0);

    const unsigned int n_trace_dofs      = fe_trace.n_dofs_per_cell();
    const unsigned int n_head_dofs       = fe_head.n_dofs_per_cell();
    const unsigned int n_flux_dofs       = fe_flux.n_dofs_per_cell();

    std::vector<types::global_dof_index> trace_dof_indices(n_trace_dofs);
    std::vector<types::global_dof_index> head_dof_indices(n_head_dofs);
    std::vector<types::global_dof_index> flux_dof_indices(n_flux_dofs);

    Vector<double> q_coeff(n_flux_dofs);

    // -----------------------------
    // Loop: locally owned cells only (MPI scalable)
    // We map the triangulation cell to the three DoFHandlers safely.
    for (auto tria_cell = triangulation.begin_active(); tria_cell != triangulation.end(); ++tria_cell) {
        if (!tria_cell->is_locally_owned())
            continue;

        //const auto flux_cell  = tria_cell->as_dof_handler_iterator(dof_handler_flux);
        //const auto head_cell  = tria_cell->as_dof_handler_iterator(dof_handler_head);
        //const auto trace_cell = tria_cell->as_dof_handler_iterator(dof_handler_trace);
        typename DoFHandler<dim>::active_cell_iterator flux_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_trace);
        typename DoFHandler<dim>::active_cell_iterator head_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_trace);
        typename DoFHandler<dim>::active_cell_iterator trace_cell(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_trace);


        flux_cell->get_dof_indices(flux_dof_indices);
        head_cell->get_dof_indices(head_dof_indices);
        trace_cell->get_dof_indices(trace_dof_indices);

        // DG0 index for this cell (one dof per cell)
        const types::global_dof_index hdof = head_dof_indices[0];

        // --------------------------------------------------------------
        // (1) head
        // --------------------------------------------------------------
        const double h = h_new[hdof];
        head_out[hdof] = h;

        // --------------------------------------------------------------
        // (2) storage rate: dS_rate = M00*(h_new-h_old)/dt
        // Need M00 per cell; read from slot cache.
        // --------------------------------------------------------------
        const unsigned int slot = tria_cell->user_index();
        Assert(slot != numbers::invalid_unsigned_int, ExcInternalError());
        local_element_data_rt_0dg0.assert_valid_slot(slot);

        const double M00 = local_element_data_rt_0dg0.get_M00(slot);

        const double dS = M00 * (h_new[hdof] - h_old[hdof]);
        const double dS_rate = dS / time_tracking.duration();
        dS_out[hdof] = dS_rate;

        // --------------------------------------------------------------
        // (3) lambda per face (6 dofs)
        // --------------------------------------------------------------
        std::array<double, GeometryInfo<dim>::faces_per_cell> lam_face;
        for (unsigned int lf = 0; lf < GeometryInfo<dim>::faces_per_cell; ++lf)
            lam_face[lf] = solution_trace[trace_dof_indices[lf]];

        if (write_lambda_to_vtu)
            for (unsigned int lf = 0; lf < GeometryInfo<dim>::faces_per_cell; ++lf)
                lambda_out[lf][hdof] = lam_face[lf];

        // --------------------------------------------------------------
        // (4) RECONSTRUCT q_h(x) AND INTEGRATE Qface[f] = ∫ q·n dΓ
        // This is the correct physical face flux for budgets and tracking.
        // --------------------------------------------------------------
        for (unsigned int i = 0; i < n_flux_dofs; ++i)
            q_coeff(i) = q_new[flux_dof_indices[i]];

        std::array<double, GeometryInfo<dim>::faces_per_cell> Qface;
        std::array<double, GeometryInfo<dim>::faces_per_cell> Af;
        Qface.fill(0.0);
        Af.fill(0.0);

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f) {
            fe_face.reinit(flux_cell, f);

            double Qf = 0.0;
            double area = 0.0;

            for (unsigned int q = 0; q < face_quad.size(); ++q) {
                area += fe_face.JxW(q);

                // Evaluate q_h at this quadrature point
                Tensor<1, dim> qh;
                for (unsigned int i = 0; i < n_flux_dofs; ++i)
                    qh += q_coeff(i) * fe_face[flux].value(i, q);

                const Tensor<1, dim> n = fe_face.normal_vector(q);
                Qf += (qh * n) * fe_face.JxW(q);
            }
            Qface[f] = Qf;
            Af[f]    = area;

            if (write_q_to_vtu)
                qface_out[f][hdof] = Qf;
            if (write_area_to_vtu)
                area_out[f][hdof] = area;
        }

        // --------------------------------------------------------------
        // (5) MNW per-cell totals:
        //     Qw_cell   = Σ link.Qe      (aquifer-well exchange rate in this cell)
        //     Qwbf_cell = Σ link.Qwbf    (wellbore flow THROUGH TOP of this cell)
        // --------------------------------------------------------------
        double Qw_sum = 0.0;
        double Qwbf_sum = 0.0;

        auto it = std::lower_bound(
            local_cell_well_map.begin(), local_cell_well_map.end(),
            trace_cell->active_cell_index(),
            [](const std::pair<unsigned int, std::vector<npsat_flow::CellWellLink>> &pair,
               unsigned int val)
            { return pair.first < val; });

        if (it != local_cell_well_map.end() && it->first == trace_cell->active_cell_index())
        {
            for (const auto &link : it->second)
            {
                Qw_sum   += link.Qe;
                //Qwbf_sum += link.Qwbf;
            }
        }

        Qw_out[hdof]   = Qw_sum;

        // --------------------------------------------------------------
        // (6) CSV row (same content/order as your tested code)
        // Note: "cell_id" in your CSV is active_cell_index(), keep that.
        // --------------------------------------------------------------
        Point<dim> cell_center = tria_cell->center();
        if (write_csv)
        {
            csv << tria_cell->id() << "," << cell_center[0] << "," << cell_center[1] << "," << cell_center[2] << ","
                << flux_dof_indices[0] << "," << flux_dof_indices[1] << "," << flux_dof_indices[2] << ","
                << flux_dof_indices[3] << "," << flux_dof_indices[4] << "," << flux_dof_indices[5] << ","
                << Qface[0] << "," << Qface[1] << "," << Qface[2] << ","
                << Qface[3] << "," << Qface[4] << "," << Qface[5] << ","
                << Qw_sum << "," << /*Qwbf_sum << "," <<*/ dS_rate << "," << h << ","
                << lam_face[0] << "," << lam_face[1] << ","
                << lam_face[2] << "," << lam_face[3] << ","
                << lam_face[4] << "," << lam_face[5]
                << "\n";
        }
    }

    if (write_csv)
        csv.close();

    // Finalize distributed vectors (owned inserts)
    head_out.compress(VectorOperation::insert);
    dS_out.compress(VectorOperation::insert);
    Qw_out.compress(VectorOperation::insert);

    if (write_q_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            qface_out[f].compress(VectorOperation::insert);

    if (write_lambda_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            lambda_out[f].compress(VectorOperation::insert);

    if (write_area_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            area_out[f].compress(VectorOperation::insert);

    // Ghosts (safe for DataOut; not strictly required for locally owned cell output,
    // but avoids surprises if DataOut touches relevant dofs)
    head_out.update_ghost_values();
    dS_out.update_ghost_values();
    Qw_out.update_ghost_values();

    if (write_q_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            qface_out[f].update_ghost_values();

    if (write_lambda_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            lambda_out[f].update_ghost_values();

    if (write_area_to_vtu)
        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
            area_out[f].update_ghost_values();

    if (write_solution_cell_vtu)
    {
        DataOut<dim> data_out;

        // IMPORTANT: attach the DG0 DoFHandler, not just the triangulation
        data_out.attach_dof_handler(dof_handler_head);

        // DG0 fields -> should become <CellData>
        data_out.add_data_vector(head_out, "head", DataOut<dim>::type_cell_data);
        data_out.add_data_vector(Qw_out,   "Qw",   DataOut<dim>::type_cell_data);
        data_out.add_data_vector(dS_out,   "dS",   DataOut<dim>::type_cell_data);


        if (write_q_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                data_out.add_data_vector(qface_out[f], "q_" + std::to_string(f),
                                         DataOut<dim>::type_cell_data);

        if (write_lambda_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                data_out.add_data_vector(lambda_out[f], "l_" + std::to_string(f),
                                         DataOut<dim>::type_cell_data);

        if (write_area_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                data_out.add_data_vector(area_out[f], "A_" + std::to_string(f),
                                         DataOut<dim>::type_cell_data);

        data_out.build_patches(/*n_subdivisions=*/1);

        std::ostringstream base;
        base << prefix << "_solution_cell_rank_" << str_rank << "_step_" << step;

        if (Utilities::MPI::n_mpi_processes(mpi_communicator) == 1)
        {
            // std::ofstream out((base.str() + ".vtk").c_str());
            // AssertThrow(out.good(), ExcMessage("Could not open VTK: " + base.str() + ".vtk"));
            // data_out.write_vtk(out);
            std::ofstream out((base.str() + ".vtu").c_str());
            data_out.write_vtu(out);
        }
        else
        {
            data_out.write_vtu_with_pvtu_record(
                "./",
                base.str(),
                time_tracking.simulation_step(),
                mpi_communicator,
                /*n_digits_for_rank=*/4,
                /*n_digits_for_cycle=*/3);
        }
    }

    if (write_cellcenters_vtk) {
        // ------------------------------------------------------------
        // Custom ASCII VTK: one POINT per locally owned cell center,
        // and all variables as POINT_DATA (no recomputation).
        // ------------------------------------------------------------
        static_assert(dim == 3, "This VTK point-cloud writer assumes dim=3.");

        const std::string step     = Utilities::int_to_string(time_tracking.simulation_step(), 3);
        const std::string str_rank = Utilities::int_to_string(my_rank, 4);

        std::ostringstream vtk_name;
        vtk_name << prefix << "_cellcenters_rank_" << str_rank << "_step_" << step << ".vtk";

        std::ofstream vtk(vtk_name.str());
        AssertThrow(vtk.good(), ExcMessage("Could not open VTK: " + vtk_name.str()));

        const unsigned int n_loc_cells = triangulation.n_locally_owned_active_cells();

        // --- Gather geometry (cell centers) in the SAME order you used for local_out_idx ---
        std::vector<Point<3>> centers;
        centers.reserve(n_loc_cells);

        // Optional: store an ID per point (useful for debugging / joining with CSV)
        std::vector<unsigned int> active_id;
        active_id.reserve(n_loc_cells);

        // Also gather data from existing vectors in the same order
        std::vector<float> head_v(n_loc_cells), dS_v(n_loc_cells), Qw_v(n_loc_cells);
        std::array<std::vector<float>, GeometryInfo<dim>::faces_per_cell> q_v;
        std::array<std::vector<float>, GeometryInfo<dim>::faces_per_cell> l_v;
        std::array<std::vector<float>, GeometryInfo<dim>::faces_per_cell> A_v;

        for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
        {
            if (write_q_to_vtu)      q_v[f].resize(n_loc_cells);
            if (write_lambda_to_vtu) l_v[f].resize(n_loc_cells);
            if (write_area_to_vtu)   A_v[f].resize(n_loc_cells);
        }

        unsigned int local_out_idx = 0;
        for (auto tria_cell = triangulation.begin_active(); tria_cell != triangulation.end(); ++tria_cell) {
            if (!tria_cell->is_locally_owned())
                continue;

            // cell center point
            centers.push_back(tria_cell->center());
            active_id.push_back(tria_cell->active_cell_index());

            // Map to head DoFHandler to read DG0 dof index (same as your DataOut block)
            //const auto head_cell_it = tria_cell->as_dof_handler_iterator(dof_handler_head);
            typename DoFHandler<dim>::active_cell_iterator head_cell_it(&triangulation, tria_cell->level(), tria_cell->index(), &dof_handler_head);

            const types::global_dof_index hdof = head_cell_it->dof_index(0);

            head_v[local_out_idx] = static_cast<float>(head_out[hdof]);
            dS_v  [local_out_idx] = static_cast<float>(dS_out  [hdof]);
            Qw_v  [local_out_idx] = static_cast<float>(Qw_out  [hdof]);

            if (write_q_to_vtu)
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    q_v[f][local_out_idx] = static_cast<float>(qface_out[f][hdof]);

            if (write_lambda_to_vtu)
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    l_v[f][local_out_idx] = static_cast<float>(lambda_out[f][hdof]);

            if (write_area_to_vtu)
                for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                    A_v[f][local_out_idx] = static_cast<float>(area_out[f][hdof]);

            ++local_out_idx;
        }

        AssertThrow(local_out_idx == n_loc_cells,
                ExcMessage("Cell-center VTK: local_out_idx mismatch. local_out_idx="
                           + std::to_string(local_out_idx) + " n_loc_cells="
                           + std::to_string(n_loc_cells)));

        // --- Write legacy VTK POLYDATA (ASCII) ---
        vtk << "# vtk DataFile Version 3.0\n";
        vtk << "Cell-center point cloud\n";
        vtk << "ASCII\n";
        vtk << "DATASET POLYDATA\n";

        vtk << "POINTS " << n_loc_cells << " float\n";
        for (unsigned int i = 0; i < n_loc_cells; ++i)
        {
            vtk << static_cast<float>(centers[i][0]) << " "
                << static_cast<float>(centers[i][1]) << " "
                << static_cast<float>(centers[i][2]) << "\n";
        }

        // One vertex cell per point (so ParaView treats them as drawable geometry)
        vtk << "VERTICES " << n_loc_cells << " " << (2 * n_loc_cells) << "\n";
        for (unsigned int i = 0; i < n_loc_cells; ++i)
            vtk << "1 " << i << "\n";

        vtk << "POINT_DATA " << n_loc_cells << "\n";

        auto write_scalar = [&](const std::string &name, const std::vector<float> &v)
        {
            vtk << "SCALARS " << name << " float 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (unsigned int i = 0; i < n_loc_cells; ++i)
                vtk << v[i] << "\n";
        };

        // Optional ID field (helps cross-check with CSV / debugging)
        {
            vtk << "SCALARS active_cell_index int 1\n";
            vtk << "LOOKUP_TABLE default\n";
            for (unsigned int i = 0; i < n_loc_cells; ++i)
                vtk << active_id[i] << "\n";
        }

        write_scalar("head", head_v);
        write_scalar("dS",   dS_v);
        write_scalar("Qw",   Qw_v);

        if (write_q_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                write_scalar("q_" + std::to_string(f), q_v[f]);

        if (write_lambda_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                write_scalar("l_" + std::to_string(f), l_v[f]);

        if (write_area_to_vtu)
            for (unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
                write_scalar("A_" + std::to_string(f), A_v[f]);

        vtk.close();
    }
}

#endif //NPSAT_FLOW_WRITE_OUTPUT_IMPL_H
