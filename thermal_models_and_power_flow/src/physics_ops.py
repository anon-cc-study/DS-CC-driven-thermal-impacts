import numpy as np

def calculate_resistance(R0, T, alpha, T0=20):
    """
    Computes the updated resistance based on temperature for a conductor.

    Parameters:
    R0 (float): Initial resistance at reference temperature (T0) in ohms.
    T (float): Current temperature in degrees Celsius.
    alpha (float): Temperature coefficient of resistance (1/°C) for the material.
    T0 (float): Reference temperature in degrees Celsius (default is 20°C).
    
    Returns:
    float: The updated resistance at temperature T.
    
    Note:
    For typical lines (aluminum and copper) the change is about 0.5%/C
    """
    return R0 * (1 + alpha * (T - T0))

# This function computes steady-state thermal line rating of overhead conductors based on IEEE std 738 heat-balance equations.
def calculate_oh_ampacity(R_T_avg, D0, Ta, Ts, H_e, Vw, epsilon, alpha, phi_degrees,
                       Lat_degrees, N, Time, Noon, Z_l_degrees, is_three_phase):
    """
    calculate_oh_ampacity

    This function computes the steady-state thermal line rating (ampacity) of overhead power line conductors based on the IEEE Std 738 heat-balance equations. It calculates the maximum allowable electric current (ampacity) such that the conductor's temperature remains within specified thermal limits, considering environmental and physical parameters.

    Parameters
    ----------
    R_T_avg : float
        Average electrical resistance of the conductor at its operating temperature [Ohms/meter].
    D0 : float
        Outside diameter of the conductor [meters].
    Ta : float
        Ambient air temperature [°C].
    Ts : float
        Surface (conductor) temperature [°C].
    H_e : float
        Elevation above sea level [meters].
    Vw : float
        Wind speed [m/s].
    epsilon : float
        Emissivity of the conductor surface [unitless, between 0 and 1].
    alpha : float
        Solar absorptivity of the conductor surface [unitless, between 0 and 1].
    phi_degrees : float
        Angle between wind direction and conductor axis [degrees].
    Lat_degrees : float
        Latitude of the conductor location [degrees].
    N : int
        Day of the year (1-365).
    Time : float
        Local solar time [hours, 0-24].
    Noon : float
        Local solar noon [hours, e.g., 12.0 for noon].
    Z_l_degrees : float
        Azimuth (orientation) angle of the conductor [degrees].
    is_three_phase : bool
        True if the conductor is part of a three-phase system (returns ampacity for all three phases), False for single-phase.

    Returns
    -------
    dict
        Dictionary with the following keys:
            'Convective heat loss (qc)': float
                Convective heat loss from the conductor [W/m].
            'Radiated heat loss (qr)': float
                Net radiated heat loss from the conductor [W/m].
            'Solar heat gain (qs)': float
                Net solar heat gain absorbed by the conductor [W/m].
            'Ampacity (I)': float
                Maximum allowable steady-state electric current (ampacity) [Amperes].

    Description
    -----------
    The function uses a physical model of heat exchange for overhead conductors. It calculates:

    - **Convective heat loss (qc):** Heat loss due to natural and forced convection (air flow), taking into account wind speed and direction.
    - **Radiated heat loss (qr):** Net heat radiated from the conductor surface to the environment.
    - **Solar heat gain (qs):** Heat absorbed due to solar radiation, computed based on the sun’s position, conductor orientation, and surface properties.
    - **Ampacity (I):** The maximum current the line can carry without exceeding the specified temperature, balancing all heat sources and sinks.

    The internal sub-functions follow the standard IEEE 738 methodology for conductor thermal analysis, accounting for both meteorological (temperature, wind, solar position) and line-specific parameters (geometry, material properties, orientation).

    Typical use: To determine the real-time or planning ampacity limit for transmission/distribution lines under varying weather and system conditions.

    References
    ----------
    - IEEE Standard 738-2012: "IEEE Standard for Calculating the Current-Temperature Relationship of Bare Overhead Conductors"
    """
    # Ampacity (I)
    def ampacity(qc, qr, qs, R_T_avg, is_three_phase=False):
        I = np.sqrt((qc + qr - qs) / R_T_avg)
        if is_three_phase:
            I *= 3  # Multiply by 3 for 3-phase lines
        return I

    # Natural convection heat loss (qc_n)
    def natural_convection(rho_f, D0, Ts, Ta):
        return 3.645 * (rho_f**0.5) * (D0**0.75) * ((Ts - Ta)**1.25)

    # Forced convection heat loss (q_c1)
    def forced_convection_low(K_angle, N_Re, k_f, Ts, Ta):
        return K_angle * (1.01 + 1.35 * N_Re**0.52) * k_f * (Ts - Ta)

    # Forced convection heat loss (q_c2)
    def forced_convection_high(K_angle, N_Re, k_f, Ts, Ta):
        return K_angle * 0.754 * N_Re**0.6 * k_f * (Ts - Ta)

    # Wind direction factor (K_angle_val)
    def K_angle(phi_degrees):
        phi = np.deg2rad(phi_degrees)  # Convert degrees to radians
        return 1.194 - np.cos(phi) + 0.194 * np.cos(2 * phi) + 0.368 * np.sin(2 * phi)

    # Reynolds number (N_Re)
    def reynolds_number(D0, rho_f, Vw, mu_f):
        return (D0 * rho_f * Vw) / mu_f

    # Dynamic viscosity of air (mu_f)
    def dynamic_viscosity(T_film):
        return (1.458e-6 * (T_film + 273)**1.5) / (T_film + 383.4)

    # Air density (rho_f)
    def air_density(H_e, T_film):
        return (1.293 - 1.525e-4 * H_e + 6.379e-9 * H_e**2) / (1 + 0.00367 * T_film)

    # Thermal conductivity of air (k_f)
    def thermal_conductivity(T_film):
        return 2.424e-2 + 7.477e-5 * T_film - 4.407e-9 * T_film**2

    # Radiated heat loss (q_r)
    def radiated_heat_loss(D0, epsilon, Ts, Ta):
        return 17.8 * D0 * epsilon * (((Ts + 273) / 100)**4 - ((Ta + 273) / 100)**4)

    # Solar heat gain (qs)
    def solar_heat_gain(alpha, Q_se, H_c_degrees, Z_c_degrees, Z_l_degrees, D0):
        H_c = np.deg2rad(H_c_degrees)  # Convert degrees to radians
        Z_c = np.deg2rad(Z_c_degrees)
        Z_l = np.deg2rad(Z_l_degrees)
        A_prime = D0 * 1  # Projected area of conductor (D0 * 1 meter)
        solar_angle = np.arccos(np.cos(H_c) * np.cos(Z_c - Z_l))
        return alpha * Q_se * np.sin(solar_angle) * A_prime

    # Solar altitude (H_c_degrees)
    def solar_altitude(Lat_degrees, delta_degrees, omega_degrees):
        Lat = np.deg2rad(Lat_degrees)      # Convert degrees to radians
        delta = np.deg2rad(delta_degrees)
        omega = np.deg2rad(omega_degrees)
        return np.rad2deg(np.arcsin(np.cos(Lat) * np.cos(delta) * np.cos(omega) + np.sin(Lat) * np.sin(delta)))  # Return in degrees

    # Solar declination (delta_degrees)
    def solar_declination(N):
        return 23.45 * np.sin(np.deg2rad((284 + N) * (360 / 365)))  # Result in degrees

    # Hour angle (omega_degrees)
    def hour_angle(Time, Noon):
        return (Time - Noon) * 15  # Result in degrees

    # Solar azimuth (Z_c_degrees)
    def solar_azimuth(omega_degrees, Lat_degrees, delta_degrees):
        omega = np.deg2rad(omega_degrees)  # Convert degrees to radians
        Lat = np.deg2rad(Lat_degrees)
        delta = np.deg2rad(delta_degrees)
        chi = np.sin(omega) / (np.sin(Lat) * np.cos(omega) - np.cos(Lat) * np.tan(delta))
        if -180 <= omega_degrees <= 0 and chi >= 0:
            return np.rad2deg(np.arctan(chi))  # Return in degrees
        elif (-180 <= omega_degrees <= 0 and chi < 0) or (0 <= omega_degrees < 180 and chi >= 0):
            return 180 + np.rad2deg(np.arctan(chi))  # Return in degrees
        else:
            return 360 + np.rad2deg(np.arctan(chi))  # Return in degrees

    # Total solar and sky radiated heat intensity corrected for elevation (Q_se_val)
    def Q_se(K_solar, Q_s):
        return K_solar * Q_s

    # Solar elevation correction factor (K_solar_val)
    def K_solar(A_tilde, B_tilde, C_tilde, H_e):
        return A_tilde + B_tilde * H_e + C_tilde * H_e**2

    # Total solar and sky radiated heat at sea level (Q_s_val)
    def Q_s(A, B, C, D, E, F, G, H_c_degrees):
        H_c = H_c_degrees
        return A + B * H_c + C * H_c**2 + D * H_c**3 + E * H_c**4 + F * H_c**5 + G * H_c**6

    # Calculations
    # -----------------------------------------------------

    # Solar declination and hour angle (results in degrees)
    delta_degrees = solar_declination(N)
    omega_degrees = hour_angle(Time, Noon)

    # Solar altitude (result in degrees)
    H_c_degrees = solar_altitude(Lat_degrees, delta_degrees, omega_degrees)

    # Solar azimuth (explicit equation, result in degrees)
    Z_c_degrees = solar_azimuth(omega_degrees, Lat_degrees, delta_degrees)

    # Solar correction and total solar radiation (using coefficients from the table)
    A_tilde, B_tilde, C_tilde = 1, 1.148e-4, -1.108e-8  # Coefficients for K_solar
    K_solar_val = K_solar(A_tilde, B_tilde, C_tilde, H_e)

    A, B, C, D, E, F, G = -42.2391, 63.8044, -1.9220, 3.46921e-2, -3.61118e-4, 1.94318e-6, -4.07608e-9
    Q_s_val = Q_s(A, B, C, D, E, F, G, H_c_degrees)

    # Total solar and sky radiated heat intensity corrected for elevation
    Q_se_val = Q_se(K_solar_val, Q_s_val)

    # Solar heat gain
    qs = solar_heat_gain(alpha, Q_se_val, H_c_degrees, Z_c_degrees, Z_l_degrees, D0)

    # Calculating film temperature
    T_film = (Ts + Ta) / 2

    # Compute air properties at film temperature
    mu_f = dynamic_viscosity(T_film)      # Dynamic viscosity of air
    rho_f = air_density(H_e, T_film)      # Air density
    k_f = thermal_conductivity(T_film)    # Thermal conductivity of air

    # Convective heat loss calculations
    qc_n = natural_convection(rho_f, D0, Ts, Ta)  # Natural convection heat loss
    N_Re = reynolds_number(D0, rho_f, Vw, mu_f)  # Reynolds number
    K_angle_val = K_angle(phi_degrees)            # Wind direction factor (in degrees)
    qc_f1 = forced_convection_low(K_angle_val, N_Re, k_f, Ts, Ta)  # Forced convection (low)
    qc_f2 = forced_convection_high(K_angle_val, N_Re, k_f, Ts, Ta)  # Forced convection (high)

    # Convective heat loss (max of natural and forced convection)
    qc = max(qc_n, qc_f1, qc_f2)

    # Radiated heat loss
    qr = radiated_heat_loss(D0, epsilon, Ts, Ta)

    # Calculate ampacity (max electric current)
    I = ampacity(qc, qr, qs, R_T_avg, is_three_phase)


    # Return results
    return {
        'Convective heat loss (qc)': qc,
        'Radiated heat loss (qr)': qr,
        'Solar heat gain (qs)': qs,
        'Ampacity (I)': I
    }