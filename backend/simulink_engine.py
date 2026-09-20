"""
RetinaSetu - Simulink Telemedicine Discrete-Event Simulator Engine
Simulates district-level screening workflows (100,000+ patients/year)
across rural Primary Healthcare Centres (PHCs).
"""
import numpy as np

def run_telemedicine_simulation(
    annual_target=100000,
    num_phcs=50,
    working_days=250,
    bandwidth_mbps=1.5,
    edge_triage=True,
    num_cloud_gpus=2,
    num_tele_doctors=3
):
    """
    Executes discrete-event Monte Carlo simulation for telemedicine workflow.
    """
    np.random.seed(42)
    daily_patients_total = annual_target / working_days
    daily_patients_per_phc = daily_patients_total / num_phcs
    working_hours_day = 7.0
    hourly_rate_district = daily_patients_total / working_hours_day

    # Sample size for Monte Carlo run
    n_samples = 5000

    # 1. Image Acquisition: smartphone + 20D coaxial alignment (mean 210s)
    acq_times = np.random.lognormal(mean=np.log(210), sigma=0.25, size=n_samples)

    # 2. Automated Quality Gate
    # ~82% pass immediately, 14% borderline enhanced, 4% recapture needed
    q_rand = np.random.uniform(0, 1, size=n_samples)
    is_recapture = q_rand > 0.96
    recapture_times = is_recapture * np.random.lognormal(mean=np.log(180), sigma=0.2, size=n_samples)

    # 3. Cellular Bandwidth Upload
    image_size_mb = 2.5
    effective_bw = np.maximum(0.2, np.random.normal(bandwidth_mbps, 0.5, size=n_samples))
    upload_times_sec = (image_size_mb * 8) / effective_bw

    if edge_triage:
        # Edge on-device TFLite screens out negative cases locally, only uploading referable/borderline cases
        upload_times_sec *= 0.45

    # 4. Cloud Inference Queue
    # GPU service time: ~1.2s per patient
    gpu_capacity_hourly = num_cloud_gpus * (3600 / 1.2)
    gpu_utilization = min(0.95, hourly_rate_district / gpu_capacity_hourly)
    gpu_queue_sec = (gpu_utilization / max(0.01, 1 - gpu_utilization)) * 1.2

    # 5. Epidemiology & Grade Distribution (India rural cohort)
    # Grade 0: 72%, Grade 1: 10%, Grade 2: 11%, Grade 3: 4%, Grade 4: 3%
    # Referable DR = Grades 2, 3, 4 (~18%)
    dr_rands = np.random.uniform(0, 1, size=n_samples)
    is_referable = dr_rands > 0.82
    num_referable = int(np.sum(is_referable))

    # 6. Tele-Ophthalmologist Review Queue (< 30s enabled by Grad-CAM explainability)
    doctor_review_sec = 30.0
    doctor_capacity_hourly = (num_tele_doctors * 3600) / doctor_review_sec
    doctor_arrival_rate = (num_referable / n_samples) * hourly_rate_district
    doctor_utilization = min(0.98, doctor_arrival_rate / doctor_capacity_hourly)
    doctor_queue_min = (doctor_utilization / max(0.02, 1 - doctor_utilization)) * (doctor_review_sec / 60.0)

    # 7. Total Turnaround Time (minutes)
    total_time_sec = (
        acq_times
        + recapture_times
        + upload_times_sec
        + gpu_queue_sec
        + (is_referable * (doctor_queue_min * 60.0 + doctor_review_sec))
    )
    total_time_min = total_time_sec / 60.0

    avg_turnaround = float(np.mean(total_time_min))
    p95_turnaround = float(np.percentile(total_time_min, 95))

    # Economic impact calculations
    cost_conventional = 2500000 # INR 25 Lakh fundus camera
    cost_retinasetu = 25000     # INR 25k smartphone + 20D coaxial lens kit
    capex_savings_inr = num_phcs * (cost_conventional - cost_retinasetu)
    cost_per_screening_inr = 145 # OpEx per screening

    # Recommended doctors to keep queue wait < 15 min (utilization < 75%)
    recommended_docs = max(1, int(np.ceil(doctor_arrival_rate / ((3600 / 30.0) * 0.75))))

    # Queue timeline histogram for chart visualization
    hist_counts, bin_edges = np.histogram(total_time_min, bins=12, range=(3, 30))
    bins_data = [
        {"range": f"{bin_edges[i]:.0f}-{bin_edges[i+1]:.0f}m", "patients": int(hist_counts[i])}
        for i in range(len(hist_counts))
    ]

    return {
        "annual_target": annual_target,
        "num_phcs": num_phcs,
        "daily_patients_district": round(daily_patients_total, 1),
        "daily_patients_per_phc": round(daily_patients_per_phc, 1),
        "referable_percentage": round((num_referable / n_samples) * 100, 1),
        "avg_turnaround_min": round(avg_turnaround, 1),
        "p95_turnaround_min": round(p95_turnaround, 1),
        "doctor_utilization_pct": round(doctor_utilization * 100, 1),
        "doctor_queue_wait_min": round(doctor_queue_min, 1),
        "recommended_doctors": recommended_docs,
        "current_doctors": num_tele_doctors,
        "capex_savings_crores": round(capex_savings_inr / 10000000, 2),
        "cost_per_screening_inr": cost_per_screening_inr,
        "bandwidth_category": "2G" if bandwidth_mbps < 0.5 else ("3G" if bandwidth_mbps < 5 else "4G/5G"),
        "latency_distribution": bins_data,
        "is_bottleneck": doctor_utilization > 0.85
    }
