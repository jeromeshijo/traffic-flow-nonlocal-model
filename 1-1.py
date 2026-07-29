import random

import numpy as np
from scipy.optimize import root

import matplotlib.pyplot as plt
from scipy.optimize import root_scalar
import os


random.seed(42)
np.random.seed(42)


class Road:

    def __init__(
        self,
        road_id,
        rho_l=None,
        rho_r=None,
        L=None,
        eta=None,
        lam=None,
        commodities=None,
    ):

        # ----------------------------
        # Identification
        # ----------------------------

        self.id = road_id

        # ----------------------------
        # Boundary densities
        # (one of them may initially be unknown)
        # ----------------------------

        self.rho_l = rho_l
        self.rho_r = rho_r

        # ----------------------------
        # Physical parameters
        # ----------------------------

        self.L = L
        self.eta = eta
        self.lam = lam

        # ----------------------------
        # Commodity distribution
        # ----------------------------

        self.commodities = commodities

    def __repr__(self):

        return (
            f"Road("
            f"id={self.id}, "
            f"rho_l={self.rho_l}, "
            f"rho_r={self.rho_r})"
        )
    

class Junction:

    def __init__(
        self,
        junction_id,
        incoming_roads,
        outgoing_roads,
        commodity_destination,
        constant_in_f=10
    ):

        # ----------------------------
        # Identification
        # ----------------------------

        self.id = junction_id

        # ----------------------------
        # Connected roads
        # ----------------------------

        self.incoming_roads = incoming_roads
        self.outgoing_roads = outgoing_roads

        # ----------------------------
        # Commodity routing
        # ----------------------------

        self.commodity_destination = commodity_destination

        # ----------------------------
        # Constants
        # ----------------------------

        self.constant_in_f = constant_in_f

        # ----------------------------
        # Buffers
        # One buffer per outgoing road
        # ----------------------------

        self.buffers = {}

        for road in self.outgoing_roads:

            self.buffers[road.id] = {

            }

    # ============================================================
    # V and W_out
    # ============================================================

    def V(self, road, W):

        return np.exp(-road.lam * W)


    def W_out_0(self, road,rho_out_l):

        K = road.rho_r * (road.eta - road.L) / road.eta

        exponent = (
            (road.lam / road.eta)
            * road.rho_r
            * road.L
        )

        log_argument = (
            rho_out_l
            * (np.exp(exponent) - 1)
            + road.rho_r
        )

        return (
            (road.lam / (road.eta ** 2))
            * (
                np.log(log_argument)
                - np.log(road.rho_r)
            )
            + K
        )
    # ============================================================
    # rho_r and rho_l
    # ============================================================

    def rho_r_function(self, road, b):

        tmp_max = max(b)

        return tmp_max




    def rho_l_function(self, road, b):

        buffer_idx = self.outgoing_roads.index(road)

        def F(rho_out_l):

            W = self.W_out_0(
                road,
                rho_out_l
            )

            return (
                rho_out_l
                - b[buffer_idx]
                * self.constant_in_f
                / self.V(road, W)
            )

        result = root_scalar(
            F,
            x0=1.0,
            method="newton",
            xtol=1e-13,
            maxiter=1000,
        )

        if not result.converged:
            raise RuntimeError(
                f"rho_l root search failed on road {road.road_id}"
            )

        return result.root



    
    
    # ============================================================
    # Riccati Solution
    # ============================================================

    def riccati(self, road, x, b):

        if road in self.incoming_roads:

            rho_l = road.rho_l
            rho_r = self.rho_r_function(road, b)

        else:

            rho_l = self.rho_l_function(road, b)
            rho_r = road.rho_r

        argument_of_exp = (
            (road.lam / road.eta)
            * rho_r
            * x
        )

        return (
            rho_r
            * rho_l
            * np.exp(argument_of_exp)
        ) / (
            rho_l
            * (
                np.exp(argument_of_exp) - 1
            )
            + rho_r
        )


    # ============================================================
    # Density Function
    # ============================================================

    def q(self, road, x, b):

        return self.riccati(road, x, b)
    

    # ============================================================
    # Flux into Buffer
    # Equation (7.3)
    # ============================================================

    def phi_in_buffer_func(self, in_road, out_road, b):

        q_in = self.q(
            in_road,
            in_road.L,
            b
        )

        V_in = self.V(
            in_road,
            self.rho_r_function(in_road, b)
        )

        commodity_fraction = 0

        for commodity, fraction in enumerate(in_road.commodities):

            if self.commodity_destination[commodity] == out_road.id:
                commodity_fraction += fraction

        return (
            commodity_fraction
            * q_in
            * V_in
        )





    # ============================================================
    # Flux out of Buffer
    # Equation (7.4)
    # ============================================================

    def phi_out_buffer_func(self, out_road, b):

        buffer_idx = self.outgoing_roads.index(out_road)

        return self.constant_in_f * b[buffer_idx]
    

    # ============================================================
    # ROOT FUNCTION
    # ============================================================


    # ============================================================
    # ODE for One Buffer
    # ============================================================

    def F_for_buffer(self, out_road, b):

        phi_in = 0

        for in_road in self.incoming_roads:

            phi_in += self.phi_in_buffer_func(
                in_road,
                out_road,
                b
            )

        phi_out = self.phi_out_buffer_func(
            out_road,
            b
        )

        return phi_in - phi_out


    # ============================================================
    # ODE System
    # ============================================================

    def F(self, b):

        F_values = []

        for out_road in self.outgoing_roads:

            F_values.append(
                self.F_for_buffer(
                    out_road,
                    b
                )
            )

        return np.array(F_values)


    # ============================================================
    # ROOT SEARCH
    # ============================================================

    def solve(self):

        print("=" * 80)
        print(f"ROOT SEARCH - Junction {self.id}")
        print("=" * 80)

        tol = 1e-6

        # Initial guess for the buffer values
        x0 = np.ones(len(self.outgoing_roads))

        result = root(
            self.F,
            x0=x0,
            tol=tol
        )

        b_star = result.x

        print("Solution:", b_star)
        print("F(sol):", result.fun)
        print("norm(F(sol)):", np.linalg.norm(result.fun))
        print("Converged:", result.success)
        print("Message:", result.message)

        if not result.success:

            print(
                "No root found, norm =",
                np.linalg.norm(result.fun),
                "\n"
            )

            raise RuntimeError(
                f"Root search failed for Junction {self.id}"
            )

        # ========================================================
        # Store buffer values
        # ========================================================

        for i, road in enumerate(self.outgoing_roads):

            self.buffers[road.id]["b"] = b_star[i]

        # ========================================================
        # Update Boundary Densities
        # ========================================================

        for road in self.incoming_roads:

            road.rho_r = self.rho_r_function(
                road,
                b_star
            )

        for road in self.outgoing_roads:

            road.rho_l = self.rho_l_function(
                road,
                b_star
            )
        
        self.b = b_star

        return b_star



def random_commodities(n):

    weights = np.random.rand(n)
    weights /= np.sum(weights)

    return weights.tolist()

def random_routing(n_commodities, outgoing_roads):

    return {
        commodity: random.choice(outgoing_roads)
        for commodity in range(n_commodities)
    }

def main():

    # ========================================================
    # Boundary Conditions
    # ========================================================

    rho_l_R1 = 0.8
    rho_r_R2 = 0.45

    # ========================================================
    # Create Roads
    # ========================================================

    n_commodities = 1

    R1 = Road(1, rho_l=rho_l_R1, L=1.0, eta=1.1, lam=1.0, commodities=random_commodities(n_commodities))
    R2 = Road(2, rho_r=rho_r_R2,L=1.0, eta=1.1, lam=1.0, commodities=random_commodities(n_commodities))

    roads = {
        1: R1,
        2: R2,}


    solve_order = [
    1
    ]

  

    # ========================================================
    # Commodity Destinations / Routing 
    # ========================================================


    commodity_destination_J1 = random_routing(n_commodities, [2])


    # ========================================================
    # Create Junctions
    # ========================================================

    J1 = Junction(
        junction_id=1,
        incoming_roads=[R1],
        outgoing_roads=[R2],
        commodity_destination=commodity_destination_J1
    )


    junctions = {

        1: J1,

    }

    # ========================================================
    # Solve Network
    # ========================================================

    for junction_id in solve_order:

        junctions[junction_id].solve()

    # ========================================================
    # Final Road Parameters
    # ========================================================

    print("\n")
    print("=" * 80)
    print("FINAL ROAD PARAMETERS")
    print("=" * 80)

    for road_id in sorted(roads):

        road = roads[road_id]

        print(f"\nRoad {road.id}")
        print("-" * 40)

        print(f"rho_l       = {road.rho_l}")
        print(f"rho_r       = {road.rho_r}")
        print(f"L           = {road.L}")
        print(f"eta         = {road.eta}")
        print(f"lambda      = {road.lam}")
        print(f"commodities = {road.commodities}")


    # ==========================================================
    # Indicator (uniform) kernel
    # gamma(s) = 1 for 0 <= s <= 1
    # ==========================================================
    def compute_W_kernel(x, rho, rho_r, L, eta):

        W = np.zeros_like(x)

        for i in range(len(x)):

            x0 = x[i]
            x_end = min(x0 + eta, L)

            mask = (x >= x0) & (x <= x_end)

            xs = x[mask]
            rs = rho[mask]

            if len(xs) > 1:
                integral = np.trapezoid(rs, xs)
            else:
                integral = 0.0

            # contribution beyond end of road
            if x0 + eta > L:

                a = L
                b = x0 + eta

                if b > a:
                    integral += rho_r * (b - a)

            W[i] = integral / eta

        return W
    # ==========================================================
    # Plot everything
    # ==========================================================
    def save_all_road_plots(roads, folder_name="Unnamed"):

        os.makedirs(folder_name, exist_ok=True)

        # ------------------------------------------------------
        # Create summary figures
        # ------------------------------------------------------
        n_roads = len(roads)
        cols = int(np.ceil(np.sqrt(n_roads)))
        rows = int(np.ceil(n_roads / cols))

        fig_density, axs_density = plt.subplots(rows, cols,
                                                figsize=(5 * cols, 4 * rows))
        fig_W, axs_W = plt.subplots(rows, cols,
                                    figsize=(5 * cols, 4 * rows))
        fig_velocity, axs_velocity = plt.subplots(rows, cols,
                                                figsize=(5 * cols, 4 * rows))

        axs_density = np.array(axs_density).reshape(-1)
        axs_W = np.array(axs_W).reshape(-1)
        axs_velocity = np.array(axs_velocity).reshape(-1)

        # ------------------------------------------------------
        # Individual road plots
        # ------------------------------------------------------
        for k, road in enumerate(roads.values()):

            road_folder = os.path.join(folder_name, f"Road_{road.id}")
            os.makedirs(road_folder, exist_ok=True)

            x = np.linspace(0, road.L, 300)

            # --------------------------------------------------
            # Density
            # --------------------------------------------------

            junction = None

            for J in junctions.values():

                if (road in J.incoming_roads or
                    road in J.outgoing_roads):

                    junction = J
                    break

            if junction is None:
                raise ValueError(f"No junction found for Road {road.id}")
            
            rho = np.array([
                junction.q(road, xi, junction.b)
                for xi in x
            ])

            plt.figure(figsize=(6, 4))
            plt.plot(x, rho, linewidth=2)
            plt.scatter([x[0], x[-1]], [rho[0], rho[-1]], zorder=5)

            plt.annotate(f"{rho[0]:.3f}",
                        (x[0], rho[0]),
                        xytext=(5, 5),
                        textcoords="offset points")

            plt.annotate(f"{rho[-1]:.3f}",
                        (x[-1], rho[-1]),
                        xytext=(-35, 5),
                        textcoords="offset points")
            plt.xlabel("x")
            plt.ylim(0, 1.1)
            plt.yticks(np.arange(0, 1.1, 0.1))
            plt.ylabel("Density")
            plt.title(f"Road {road.id} Density")
            plt.grid(True)
            plt.tight_layout()

            plt.savefig(os.path.join(road_folder, "density.png"))

            plt.close()

            # Add to summary figure
            axs_density[k].plot(x, rho, linewidth=2)
            axs_density[k].scatter([x[0], x[-1]], [rho[0], rho[-1]], zorder=5)

            axs_density[k].annotate(f"{rho[0]:.3f}",
                                    (x[0], rho[0]),
                                    xytext=(5, 5),
                                    textcoords="offset points")

            axs_density[k].annotate(f"{rho[-1]:.3f}",
                                    (x[-1], rho[-1]),
                                    xytext=(-35, 5),
                                    textcoords="offset points")
            axs_density[k].set_title(f"Road {road.id}")
            axs_density[k].set_ylim(0, 1.1)
            axs_density[k].set_yticks(np.arange(0, 1.1, 0.1))
            axs_density[k].set_xlabel("x")
            axs_density[k].set_ylabel("Density")
            axs_density[k].grid(True)

            # --------------------------------------------------
            # Nonlocal density
            # --------------------------------------------------
            W = compute_W_kernel(
                x,
                rho,
                road.rho_r,
                road.L,
                road.eta
            )

            plt.figure(figsize=(6, 4))
            plt.scatter([x[0], x[-1]], [W[0], W[-1]], zorder=5)

            plt.annotate(f"{W[0]:.3f}",
                        (x[0], W[0]),
                        xytext=(5, 5),
                        textcoords="offset points")

            plt.annotate(f"{W[-1]:.3f}",
                        (x[-1], W[-1]),
                        xytext=(-35, 5),
                        textcoords="offset points")
            plt.plot(x, W, linewidth=2)
            
            plt.xlabel("x")
            plt.ylabel("Nonlocal Density")
            plt.ylim(0, 2)
            plt.yticks(np.arange(0, 2.01, 0.2))
            plt.title(f"Road {road.id} Nonlocal Density")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(road_folder, "nonlocal_density.png"))
            plt.close()

            # Add to summary figure
            axs_W[k].plot(x, W, linewidth=2)
            axs_W[k].scatter([x[0], x[-1]], [W[0], W[-1]], zorder=5)

            axs_W[k].annotate(f"{W[0]:.3f}",
                            (x[0], W[0]),
                            xytext=(5, 5),
                            textcoords="offset points")

            axs_W[k].annotate(f"{W[-1]:.3f}",
                            (x[-1], W[-1]),
                            xytext=(-35, 5),
                            textcoords="offset points")
            axs_W[k].set_title(f"Road {road.id}")
            axs_W[k].set_xlabel("x")
            axs_W[k].set_ylim(0, 2)
            axs_W[k].set_yticks(np.arange(0, 2.01, 0.2))
            axs_W[k].set_ylabel("Nonlocal Density")
            axs_W[k].grid(True)

            # --------------------------------------------------
            # Velocity
            # --------------------------------------------------
            V = np.exp(-road.lam * W)



            plt.figure(figsize=(6, 4))

            plt.scatter([x[0], x[-1]], [V[0], V[-1]], zorder=5)

            plt.annotate(f"{V[0]:.3f}",
                        (x[0], V[0]),
                        xytext=(5, 5),
                        textcoords="offset points")

            plt.annotate(f"{V[-1]:.3f}",
                        (x[-1], V[-1]),
                        xytext=(-35, 5),
                        textcoords="offset points")
            plt.plot(x, V, linewidth=2)
            plt.xlabel("x")
            plt.ylim(0, 1)
            plt.yticks(np.arange(0, 1.01, 0.1))
            plt.ylabel("Velocity")
            plt.title(f"Road {road.id} Velocity")
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(road_folder, "velocity.png"))
            plt.close()

            # Add to summary figure
            axs_velocity[k].plot(x, V, linewidth=2)

            axs_velocity[k].scatter([x[0], x[-1]], [V[0], V[-1]], zorder=5)

            axs_velocity[k].annotate(f"{V[0]:.3f}",
                                    (x[0], V[0]),
                                    xytext=(5, 5),
                                    textcoords="offset points")

            axs_velocity[k].annotate(f"{V[-1]:.3f}",
                                    (x[-1], V[-1]),
                                    xytext=(-35, 5),
                                    textcoords="offset points")
            axs_velocity[k].set_title(f"Road {road.id}")
            axs_velocity[k].set_ylim(0, 1)
            axs_velocity[k].set_yticks(np.arange(0, 1.01, 0.1))
            axs_velocity[k].set_xlabel("x")
            axs_velocity[k].set_ylabel("Velocity")
            axs_velocity[k].grid(True)

        # ------------------------------------------------------
        # Remove unused subplots
        # ------------------------------------------------------
        for i in range(n_roads, len(axs_density)):
            fig_density.delaxes(axs_density[i])
            fig_W.delaxes(axs_W[i])
            fig_velocity.delaxes(axs_velocity[i])

        # ------------------------------------------------------
        # Save summary figures
        # ------------------------------------------------------
        fig_density.suptitle("Density Profiles of All Roads", fontsize=16)
        fig_density.tight_layout(rect=[0, 0, 1, 0.96])
        fig_density.savefig(os.path.join(folder_name,
                                        "density_all_roads.png"))
        plt.close(fig_density)

        fig_W.suptitle("Nonlocal Density Profiles of All Roads", fontsize=16)
        fig_W.tight_layout(rect=[0, 0, 1, 0.96])
        fig_W.savefig(os.path.join(folder_name,
                                "nonlocal_density_all_roads.png"))
        plt.close(fig_W)

        fig_velocity.suptitle("Velocity Profiles of All Roads", fontsize=16)
        fig_velocity.tight_layout(rect=[0, 0, 1, 0.96])
        fig_velocity.savefig(os.path.join(folder_name,
                                        "velocity_all_roads.png"))
        plt.close(fig_velocity)


    # ==========================================================
    # Create all plots
    # ==========================================================
    save_all_road_plots(
        roads,
        "1-1"
    )

if __name__ == "__main__":
    main()

