import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import root
from pathlib import Path
import utils

IMAGES_DIR = Path(__file__).resolve().parent / "images"


def _finalize_figure(fig, show=True, save_path=None, dpi=200):
    """Save and/or show a figure; close it if not shown interactively."""
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close(fig)


class Crossroad:
    id_counter = 0
    crossroad_list = []

    def __init__(self):
        self.predecessors = []
        self.streets_incoming = [] 
        self.successors = []
        self.streets_outgoing = []
        self.n = len(self.predecessors)
        self.m = len(self.successors)
        self.is_origin = self.n == 0            # is true if has no predecessors => is origin
        self.is_destination = self.m == 0       # is true if has no successors => is destination
        self.probabilities = None
        self.id = Crossroad.id_counter          # id is the index of the crossroad
        Crossroad.id_counter += 1

        Crossroad.crossroad_list.append(self)

    # with cascade we add 'self' as a successor to the predecessor
    # without cascade we have to be careful, because streets could be used multiple times
    def add_predecessor(self, predecessor: "Crossroad", street_template: "Street", cascade: bool = True):
        self.predecessors.append(predecessor)
        if cascade:
            street = street_template.create_street(cr_begin=predecessor, cr_end=self)
        else:   # if not cascade, use the same street
            street = street_template
        self.streets_incoming.append(street)
        self.n = len(self.predecessors)
        self.is_origin = self.n == 0
        if cascade:
            predecessor.add_successor(self, street, cascade=False)

    # with cascade we add 'self' as a successor to the predecessor
    # without cascade we have to be careful, because streets could be used multiple times
    def create_and_add_predecessor(self, street: "Street", cascade: bool = True): 
        predecessor = Crossroad()
        self.add_predecessor(predecessor, street, cascade=True)
        return predecessor

    # with cascade we add 'self' as a predecessor to the successor
    # without cascade we have to be careful, because streets could be used multiple times
    def add_successor(self, successor: "Crossroad", street_template: "Street", cascade: bool = True):
        self.successors.append(successor)
        if cascade:
            street = street_template.create_street(cr_begin=self, cr_end=successor)
        else:   # if not cascade, use the same street
            street = street_template
        self.streets_outgoing.append(street)
        self.m = len(self.successors)
        self.is_destination = self.m == 0
        if cascade:
            successor.add_predecessor(self, street, cascade=False)

    # with cascade we add 'self' as a predecessor to the successor
    # without cascade we have to be careful, because streets could be used multiple times
    def create_and_add_successor(self, street: "Street", cascade: bool = True): 
        successor = Crossroad()
        self.add_successor(successor, street, cascade=True)
        return successor

    def set_probabilities_using_commodities(self, c):
        if not self.is_destination:
            self.probabilities = 1 / self.m * np.ones(self.m)
        # TODO this ist just quick and dirty for 1tom
        # also functions for n to m

    def get_probability(self, street):
        index = self.streets_outgoing.index(street)
        return self.probabilities[index]

    def solve_this_crossroad(self):
        pass  # something with root F

    def print_info(self):
        print(f"Crossroad {self.id}", end="")
        if self.is_origin and self.is_destination:
            print("  (lonely crossroad without connections) ")
        if self.is_origin:
            print("  (origin) ")
        elif self.is_destination:
            print("  (destination) ")
        else:
            print("  (inner crossroad) ")
        print(f"  Dimensions: {self.n} x {self.m}")
        print(f"  Predecessors: {self.predecessors}")
        print(f"    Streets from predecessors: {self.streets_incoming}")
        print(f"  Successors: {self.successors}")
        print(f"    Streets to successors: {self.streets_outgoing}")
        print()

    @staticmethod
    def clear_list():
        Crossroad.crossroad_list.clear()

    @staticmethod
    def print_list():   
        raise ValueError("This method is deprecated. Please use funktion 'print_crossroad_list' from class 'Network'. ")
        # print("=" * 80)
        # if len(Crossroad.crossroad_list) < 1:
        #     print("Crossroad list is empty. ")
        # for cr in Crossroad.crossroad_list:
        #     cr.print_info()
        # print("=" * 80)

    def __str__(self):
        return f"Crossroad {self.id}"

    def __repr__(self):
        return self.__str__()

# end of class Crossroad


# a street cannot be used twice. (wouldn't make sense)
# But a instance of this class is more like a template for a street. 
# So if i write 'myFirstCR.create_and_add_predecessor(s1, cascade=True)', 
# that means that a street with the parameters of s1 is created and placed between those crossroads. 
# So I can use s1 multiple times without problem
class Street:
    id_counter = 0
    street_list = []

    # streets are always one directional. If you want a two directional street, you have to add 2 seperate ones. 
    # street instances only get a id, if the are a real street being created with create_street
    # streets created via constructor directly are just street templates (without id, begin and end crs)
    def __init__(self, params, constant_in_f=10):
        # params is a dictionary with the parameters of the crossroad: lambda, eta, L, q_max, rho_l, rho_r
        self.params = params
        self.constant_in_f = constant_in_f
        self.begin = None
        self.end = None
        self.id = None

    def create_street(self, cr_begin, cr_end):
        street = Street(self.params, self.constant_in_f)
        street.begin = cr_begin
        street.end = cr_end
        street.id = Street.id_counter  # id is the index of the crossroad
        Street.id_counter += 1

        Street.street_list.append(street)
        return street

    # this is the velocity function V(W). 
    # its in this class, because every street has its own V. 
    # and we don't have to keep track of in/out street, or which street were calculating it on
    def V(self, w):
        return np.exp(-self.params['lambda'] * w)

    def get_global_id(self): 
        return self.id

    # this function calcs rho_r for street 'self'
    def rho_r_func(self, b): 
        cr = self.end

        # max_b_value is not set to 0, because that would violate a assumption in riccati
        max_b_value = 1e-6

        if cr.is_destination:
            return utils.rho_r_out

        for street in cr.streets_outgoing:
            p = cr.get_probability(street)
            if p != 0:                
                global_id = street.get_global_id()
                max_b_value = max(max_b_value, b[global_id]) 

        return max_b_value

    def rho_l_expr(self, b, w_0):
        """
        Computes rho_l as a function of b and a given w_0 value.
        Unlike rho_l_func, this does NOT call self.W_0(b) internally —
        w_0 is passed in as a parameter instead, breaking the circular
        dependency so the root solver can vary it freely.
        """        

        if self.begin.is_origin: 
            return utils.rho_l_in

        global_id = self.get_global_id()
        return b[global_id] * self.constant_in_f / self.V(w_0)

    def W_0_expr(self, b, rho_l):
        """
        Computes W_0 as a function of b and a given rho_l value.
        Unlike W_0, this does NOT call self.rho_l_func(b) internally —
        rho_l is passed in as a parameter instead, breaking the circular
        dependency so the root solver can vary it freely.
        """
        lambda_, eta, L, q_max = self.params.values()
        rho_r = self.rho_r_func(b)
        K = rho_r * (eta - L) / eta

        with np.errstate(invalid='raise'):
            result = np.log(rho_l * (np.exp((lambda_ / eta) * rho_r * L) - 1) + rho_r)

        return (
            (lambda_ / (eta ** 2))
            * (
                np.log(rho_l * (np.exp((lambda_ / eta) * rho_r * L) - 1) + rho_r)
                - np.log(rho_r)
            )
            + K
        )

    # rho_l(q(0)) and W_0(W(0)) depend on each other. We have 2 equations and 2 unknowns, I tried to solve it analytically but failed. 
    # So in this function im solving it numerically
    def calculate_rho_l_and_W_0_old(self, b):
        """
        rho_l and W_0 depend on each other (rho_l needs W_0, W_0 needs rho_l),
        so this is a fixed-point problem, not something to compute directly.
        We solve it numerically: find (w_0, rho_l) such that both equations
        hold simultaneously.
        """
        def equations(vars):
            w_0, rho_l = vars
            # rho_l equation: 
            eq1 = rho_l - self.rho_l_expr(b, w_0)
            # w_0 equation:
            eq2 = w_0 - self.W_0_expr(b, rho_l)
            return [eq1, eq2]

        counter = 0
        while (True):
            min_val = 0
            max_val = 100
            x0 = np.random.uniform(low=min_val, high=max_val, size=2)       
            result = root(equations, x0)  # , method='lm')  # , options={'maxiter': 10000})

            if result.x[0] < 0 or result.x[1] < 0:
                continue

            counter += 1
            if result.success or counter > 100:
                break

        # print("Results after loop: ")
        # print("counter: ", counter)
        # print("startpoint: ", x0)

        if not result.success:
            print("Message: ", result.message)
            raise ValueError("root has not found a solution for rho_l and W_0")

        # print("Solution found! ")
        # print("Solution: w_0 =", result.x[0], "|| rho_l =", result.x[1])
        # print("Message:", result.message)
        return result.x

    def calculate_rho_l_and_W_0(self, b):
        """
        rho_l and W_0 depend on each other (rho_l needs W_0, W_0 needs rho_l),
        so this is a fixed-point problem, not something to compute directly.

        Deterministic version: use Jerome fixed-point iteration for rho_l,
        then evaluate W_0 from the closed-form expression. This avoids random
        restarts so F(b) is a well-defined function for the outer root finder.
        """

        # Origin streets use a prescribed density rho_l = 1 (independent of b).
        if self.begin.is_origin:
            rho_l = utils.rho_l_in
        else:
            rho_l = self.jerome_rho_l_function(b)

        w_0 = self.W_0_expr(b, rho_l)
        return np.array([w_0, rho_l])

    def jerome_rho_l_function(self, b):
        # Origin streets do not use a free buffer; rho_l is fixed.
        if self.begin.is_origin:
            return utils.rho_l_in

        buffer_idx = self.get_global_id()

        # Initial guess: assume W = 0, so V = 1
        rho_l = (
            b[buffer_idx]
            * self.constant_in_f
        )

        tol = 1e-13
        max_iter = 100

        for i in range(max_iter):

            W = self.W_0_expr(
                b, 
                rho_l
            )

            rho_new = (
                b[buffer_idx]
                * self.constant_in_f
                / self.V(W)
            )

            if abs(rho_new - rho_l) < tol:
                # print(i)
                return rho_new

            rho_l = rho_new

        return rho_l

    def q(self, x, b): 
        # riccati function is imported from self created 'utils.py'
        w_0, rho_l = self.calculate_rho_l_and_W_0(b)
        return utils.riccati(
            rho_l=rho_l, 
            rho_r=self.rho_r_func(b), 
            lambda_=self.params['lambda'], 
            eta=self.params['eta'], 
            x=x
        )

    def W_on_grid(self, x, q_values, rho_r):
        """
        Steady-state nonlocal impact W on a spatial grid.

        Uses the finite-street extension of the density beyond L by rho_r
        (same formula as in the older compute_W helpers):
            W(x) = (int_x^L q(y) dy + rho_r * (x + eta - L)) / eta
        """
        L = self.params['L']
        eta = self.params['eta']
        W = np.zeros(len(x), dtype=float)
        for i in range(len(x)):
            W[i] = (
                np.trapezoid(q_values[i:], x[i:])
                + rho_r * (x[i] + eta - L)
            ) / eta
        return W

    def steady_state_profiles(self, b, n_points=200):
        """
        Compute steady-state profiles on [0, L] for this street.

        Returns
        -------
        x, q_values, W, v : ndarrays of shape (n_points,)
        """
        L = self.params['L']
        eta = self.params['eta']
        lambda_ = self.params['lambda']

        x = np.linspace(0.0, L, n_points)
        _w_0, rho_l = self.calculate_rho_l_and_W_0(b)
        rho_r = self.rho_r_func(b)

        # Pointwise evaluation: utils.riccati uses `if x == 0`, so no array x.
        q_values = np.array(
            [utils.riccati(rho_l, rho_r, lambda_, eta, xi) for xi in x],
            dtype=float,
        )
        W = self.W_on_grid(x, q_values, rho_r)
        v = self.V(W)
        return x, q_values, W, v

    def travel_time(self, b, n_points=200):
        """
        Steady-state travel time of this street (thesis Thm. 9.4 / eq. (9.1)):

            DALETH_s = int_0^L 1 / v(x) dx,   v(x) = V(W(x))

        Returns np.inf if the velocity vanishes somewhere on the street.
        """
        x, _q, _W, v = self.steady_state_profiles(b, n_points=n_points)

        if np.any(v <= 0):
            return np.inf

        return float(np.trapezoid(1.0 / v, x))

    def print_info(self):
        if self.id is None: 
            print("This street is just a street_template, no concrete street. ")
            print(f"  params: {self.params}")
            print()
            return
        print(f"Street {self.id}")
        print(f"  begin: {self.begin}")
        print(f"  end: {self.end}")
        print(f"  params: {self.params}")
        print()

    @staticmethod
    def clear_list():
        Street.street_list.clear()

    @staticmethod
    def print_list():    
        raise ValueError("This method is deprecated. Please use function 'print_street_list' from class 'Network'. ")
        # print("=" * 80)
        # if len(Street.street_list) < 1:
        #     print("Street list is empty. ")
        # for s in Street.street_list:
        #     s.print_info()
        # print("=" * 80)

    def __str__(self):
        return f"Street {self.id}"

    def __repr__(self):
        return self.__str__()
# end of class Street


class Commodity:
    def __init__(self, path: list[Crossroad]):
        self.path = path 


# this class is the network an has the functions to solve it
class Network:
    def __init__(self, crossroad_list: list[Crossroad], street_list: list[Street], commodity_list: list[Commodity] = None):
        self.crossroad_list = list(crossroad_list)
        self.street_list = list(street_list)
        if commodity_list is not None:
            self.commodity_list = list[Commodity](commodity_list)
        else: 
            self.commodity_list = None
        self.number_of_streets = len(self.street_list)

        # Free buffer variables are only for streets that do not leave an origin.
        # Origin streets use a prescribed rho_l_in in utils.py, so their buffer is not a
        # degree of freedom and must not appear in the outer root problem.
        self.buffer_streets = [
            street for street in self.street_list if not street.begin.is_origin
        ]
        self.number_of_buffers = len(self.buffer_streets)

        self.set_all_probs_for_all_crs()

        assert self.check_all_commodity_paths()

        clear_crossroad_list_and_street_list()

    def set_all_probs_for_all_crs(self):
        for cr in self.crossroad_list:
            cr.set_probabilities_using_commodities(0)

    # returns True if ALL commodities are valid, else returns False
    def check_all_commodity_paths(self):
        if self.commodity_list is None: 
            return True
        for c in self.commodity_list:
            if not self.check_commodity_path(c):
                return False
        return True

    def check_commmodity_path(self, c: Commodity):

        if not c.path[0].is_origin:
            return False

        for i in range(len(c.path)):
            # if cr is last in path
            if i == len(c.path) - 1:
                # if last cr is not a destination
                if not c.path[i].is_destination:
                    return False
                else:
                    return True

            # if next cr in path is not reachable: 
            if not c.path[i + 1] in c.path[i].successors:
                return False

    def expand_buffers(self, x):
        """
        Map the free-buffer vector x (length = number_of_buffers) to a full
        buffer array b indexed by street.id (length = number_of_streets).
        Origin-street slots stay 0 and are unused (rho_l is fixed to 1 there).
        """
        b = np.zeros(self.number_of_streets, dtype=float)
        for i, street in enumerate(self.buffer_streets):
            b[street.get_global_id()] = x[i]
        return b

    # this function calcs the flux of all streets going into 'street'. 
    def phi_in_street_func(self, street: Street, b):
        cr = street.begin

        phi_in = 0
        for street_in in cr.streets_incoming:

            L = street_in.params['L']
            q = street_in.q(x=L, b=b)
            V = street_in.V(street_in.rho_r_func(b))

            phi_in += cr.get_probability(street) * q * V

        return phi_in

    # this function calcs the flux of the buffer into 'street' => "out of buffer"
    def phi_out_street_func(self, street: Street, b):

        global_id = street.get_global_id()
        return street.constant_in_f * b[global_id]

        # # the following is equivalent, but weird to compute because rho_l and W_0. 
        # q = street.rho_l_func(b)
        # V = street.V(street.W_out_0(b))
        # return q * V

    # this function is for one street => return one value (that we want to be 0)
    def F_for_street(self, street, b):        
        phi_in = self.phi_in_street_func(street, b)
        phi_out = self.phi_out_street_func(street, b)
        return (
            phi_in
            -
            phi_out
        ) 

    def F_for_network(self, x):
        """
        Residual vector for the free buffers only (non-origin streets).
        x has length number_of_buffers; return value has the same length.
        """
        # --- old version: one residual per street, origin residual forced to 0 ---
        # f_array = np.zeros_like(self.street_list)
        # counter = 0
        # for street in self.street_list:
        #     # if street comes from origin, f is perfectly fulfilled already
        #     if street.begin.is_origin:
        #         f_array[counter] = 0
        #     else:
        #         f_array[counter] = self.F_for_street(street, b)
        #     counter += 1
        # f_array = np.asarray(f_array, dtype=float)
        # return f_array

        b = self.expand_buffers(x)
        f_array = np.zeros(self.number_of_buffers, dtype=float)
        for i, street in enumerate(self.buffer_streets):
            f_array[i] = self.F_for_street(street, b)
        return f_array

    def solve(self):
        x0 = np.ones(self.number_of_buffers)

        result = root(self.F_for_network, x0=x0, tol=1e-13)

        b_star = self.expand_buffers(result.x)

        if not result.success:
            print("x_star (free buffers):", result.x)
            print("F(sol):", result.fun)    
            print("norm(F(sol)):", np.linalg.norm(result.fun))
            print("Converged:", result.success)
            print("Message:", result.message)

        return b_star

    def street_travel_times(self, b, n_points=200):
        """Travel time of every street in the network (thesis eq. (9.1))."""
        return {
            street.id: street.travel_time(b, n_points=n_points)
            for street in self.street_list
        }

    def path_travel_time(self, streets, b, n_points=200):
        """
        Steady-state path travel time (thesis eq. (9.2)):
        sum of the travel times of the streets along the path.
        """
        return float(sum(
            street.travel_time(b, n_points=n_points) for street in streets
        ))

    def print_crossroad_list(self):   
        print("=" * 80)
        if len(self.crossroad_list) < 1:
            print("Crossroad list is empty. ")
        for cr in self.crossroad_list:
            cr.print_info()
        print("=" * 80)

    def print_street_list(self):   
        print("=" * 80)
        if len(self.street_list) < 1:
            print("Street list is empty. ")
        for s in self.street_list:
            s.print_info()
        print("=" * 80)


def clear_crossroad_list_and_street_list():
    Crossroad.clear_list()
    Crossroad.id_counter = 0
    Street.clear_list()
    Street.id_counter = 0


# Automatically builds a rectangular-grid network.
# Each crossroad is connected to the neighbor below and to the right, if it exists.
def build_rowsxcols_network_auto(rows, cols):
    params = utils.default_params
    s0 = Street(params)

    grid = np.empty((rows, cols), dtype=object)

    # fill grid with crossroads
    for i in range(rows): 
        for j in range(cols): 
            grid[i, j] = Crossroad()

    for i in range(rows):
        for j in range(cols):
            # street downward (next row):
            if i != rows - 1:
                grid[i, j].add_successor(grid[i + 1, j], s0)
            # street to the right (next column):
            if j != cols - 1:
                grid[i, j].add_successor(grid[i, j + 1], s0)

    network = Network(Crossroad.crossroad_list, Street.street_list)

    return network


def build_braess_network(with_additional_street: bool):
    params = utils.default_params
    s0 = Street(params)
    s_additional = Street(utils.params_for_short_street)

    cr0 = Crossroad()
    cr1 = cr0.create_and_add_successor(s0)
    cr2 = cr1.create_and_add_successor(s0)
    cr3 = cr1.create_and_add_successor(s0)

    cr4 = cr3.create_and_add_successor(s0)
    cr2.add_successor(cr4, s0)

    cr4.create_and_add_successor(s0)

    if with_additional_street:
        cr2.add_successor(cr3, s_additional)

    network = Network(Crossroad.crossroad_list, Street.street_list)

    return network


def create_m_following_streets_recursive(cr: Crossroad, street: Street, m: int, counter: int):
    if counter == 1:
        return
    for _ in range(m):
        crm = cr.create_and_add_successor(street)
        create_m_following_streets_recursive(crm, street, m, counter - 1)


def build_network_with_only_1xm_crossroads(cols: int, m: int):
    params = {
        'lambda': 1.0,
        'eta': 1.0,
        'L': 1.0,
        'q_max': 1.0  # ,
        # 'rho_l': 1.0,
        # 'rho_r': 1e-6
    }
    s0 = Street(params, constant_in_f=10)

    cr0 = Crossroad()

    create_m_following_streets_recursive(cr0, s0, m, cols)

    network = Network(Crossroad.crossroad_list, Street.street_list)

    return network


def _build_graph(network: Network):
    """Build a NetworkX digraph from Crossroad/Street objects."""
    G = nx.DiGraph()
    for cr in network.crossroad_list:
        G.add_node(cr.id, obj=cr)
    for street in network.street_list:
        if street.begin is not None and street.end is not None:
            G.add_edge(street.begin.id, street.end.id, obj=street)
    return G


# ============================================================
# Fixed grid layout (e.g. 4x4) for regular networks
# ============================================================
def draw_network_grid(cols, rows, network: Network, id_order=None, spacing=2.0,
                      show=True, save_path=None):
    """
    Place nodes on a fixed grid instead of using a layout algorithm.

    cols, rows: number of columns/rows of the grid
    id_order:   list of cr.id in the order they should be placed on the grid
                (row by row, left -> right, top -> bottom).
                If None, use Crossroad ids sorted by id.
    spacing:    distance between neighboring nodes
    """
    G = _build_graph(network)

    if id_order is None:
        id_order = sorted(cr.id for cr in network.crossroad_list)

    if len(id_order) != cols * rows:
        raise ValueError(f"cols*rows ({cols * rows}) does not match the number "
                         f"of nodes ({len(id_order)})")

    # Fixed positions: row by row
    pos = {}
    for idx, node_id in enumerate(id_order):
        col = idx % cols
        row = idx // cols
        pos[node_id] = (col * spacing, -row * spacing)  # -row: top to bottom

    fig, ax = plt.subplots(figsize=(cols * 1.8, rows * 1.8))

    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='deepskyblue', node_size=950,
                           edgecolors='black', linewidths=2)
    labels = {cr.id: f"CR{cr.id}" for cr in network.crossroad_list}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=13, font_color='navy', font_weight='bold')

    nx.draw_networkx_edges(G, pos, ax=ax, edge_color='grey', arrows=True,
                           arrowstyle='-|>', width=2.2, arrowsize=25,
                           connectionstyle='arc3,rad=0')  # rad was 0.08

    edge_labels = {(s.begin.id, s.end.id): f"S{s.id}"
                   for s in network.street_list if s.begin and s.end}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, ax=ax,
                                 font_color="darkgreen", font_size=10, font_weight="bold")

    ax.set_aspect('equal')  # important: otherwise the grid looks distorted
    ax.axis('off')
    fig.tight_layout(pad=1.5)
    _finalize_figure(fig, show=show, save_path=save_path)
    return fig, ax


def draw_network_grid_colored(
    cols,
    rows,
    network: Network,
    b=None,
    metric='buffer',
    id_order=None,
    spacing=2.0,
    edge_values=None,
    edge_colors=None,
    n_points=200,
    cmap_name='viridis',
    default_color='lightgrey',
    show=True,
    save_path=None,
):
    """
    Same fixed grid layout as draw_network_grid, but colors streets.

    Color modes (first match wins):
      1) edge_colors: dict {street.id: color}  -- hardcode which streets share a color
      2) edge_values: dict {street.id: float}  -- continuous colormap
      3) metric + b:  'buffer' | 'travel_time' | 'flux'

    Example hardcoding:
      edge_colors = {
          0: 'C0', 1: 'C0', 2: 'C0',   # same color
          5: 'C1', 8: 'C1',
          6: 'C2',
      }
    Streets missing from edge_colors use default_color.
    """
    G = _build_graph(network)

    if id_order is None:
        id_order = sorted(cr.id for cr in network.crossroad_list)

    if len(id_order) != cols * rows:
        raise ValueError(f"cols*rows ({cols * rows}) does not match number "
                         f"of nodes ({len(id_order)})")

    pos = {}
    for idx, node_id in enumerate(id_order):
        col = idx % cols
        row = idx // cols
        pos[node_id] = (col * spacing, -row * spacing)

    edges = []
    streets_drawn = []
    for street in network.street_list:
        if street.begin is None or street.end is None:
            continue
        edges.append((street.begin.id, street.end.id))
        streets_drawn.append(street)

    if len(edges) == 0:
        raise ValueError('no streets to draw')

    fig, ax = plt.subplots(figsize=(cols * 1.8, rows * 1.8))

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color='deepskyblue', node_size=950,
        edgecolors='black', linewidths=2,
    )
    labels = {cr.id: f'CR{cr.id}' for cr in network.crossroad_list}
    nx.draw_networkx_labels(
        G, pos, labels, ax=ax, font_size=13, font_color='navy', font_weight='bold',
    )

    use_hardcoded = edge_colors is not None

    if use_hardcoded:
        for (u, w), street in zip(edges, streets_drawn):
            color = edge_colors.get(street.id, default_color)
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, w)], ax=ax, edge_color=[color],
                width=3.0, arrows=True, arrowstyle='-|>', arrowsize=25,
                connectionstyle='arc3,rad=0',
            )
        # title = 'Grid network with hardcoded street colors'
    else:
        values = []
        for street in streets_drawn:
            if edge_values is not None:
                values.append(float(edge_values[street.id]))
            else:
                if b is None:
                    raise ValueError('b is required when not using edge_colors')
                if metric == 'buffer':
                    values.append(float(b[street.get_global_id()]))
                elif metric == 'travel_time':
                    values.append(float(street.travel_time(b, n_points=n_points)))
                elif metric == 'flux':
                    values.append(float(network.phi_out_street_func(street, b)))
                else:
                    raise ValueError("metric must be 'buffer', 'travel_time' or 'flux'")

        values = np.asarray(values, dtype=float)
        vmin, vmax = float(np.min(values)), float(np.max(values))
        if vmin == vmax:
            vmax = vmin + 1e-12

        cmap = plt.get_cmap(cmap_name)
        for (u, w), val in zip(edges, values):
            color = cmap((val - vmin) / (vmax - vmin))
            nx.draw_networkx_edges(
                G, pos, edgelist=[(u, w)], ax=ax, edge_color=[color],
                width=3.0, arrows=True, arrowstyle='-|>', arrowsize=25,
                connectionstyle='arc3,rad=0',
            )

        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
        sm.set_array([])
        cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(metric if edge_values is None else 'edge value')
        # title = f'Grid network colored by {metric if edge_values is None else "edge_values"}'

    edge_labels = {
        (s.begin.id, s.end.id): f'S{s.id}'
        for s in network.street_list if s.begin and s.end
    }
    nx.draw_networkx_edge_labels(
        G, pos, edge_labels=edge_labels, ax=ax,
        font_color='black', font_size=10, font_weight='bold',
    )

    #     ax.set_title(title)
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout(pad=1.5)
    _finalize_figure(fig, show=show, save_path=save_path)
    return fig, ax


# ============================================================
# Braess-paradox layout
# CR left -- CR -- (split up/down) -- merge -- CR right
# ============================================================
def draw_network_braess(network: Network, id_order=None, spacing=2.0):
    """
    Draw the network in Braess-paradox layout:

        CR2
       /    \\
    CR0--CR1  CR4--CR5
       \\    /
        CR3

    id_order: list of 6 cr.id in the order
              [left, middle, upper, lower, merge, right].
              If None, use sorted Crossroad ids.
    spacing:  distance between neighboring nodes
    """
    G = _build_graph(network)

    if id_order is None:
        id_order = sorted(cr.id for cr in network.crossroad_list)

    if len(id_order) != 6:
        raise ValueError(f"Braess layout needs exactly 6 nodes, "
                         f"but there are {len(id_order)}")

    # fixed positions: left, middle, split (up/down), merge, right
    offsets = [
        (0, 0),   # CR left
        (1, 0),   # CR next to it on the right
        (2, 1),   # diagonally up-right
        (2, -1),  # diagonally down-right
        (3, 0),   # merge
        (4, 0),   # CR further right
    ]
    pos = {
        node_id: (ox * spacing, oy * spacing)
        for node_id, (ox, oy) in zip(id_order, offsets)
    }

    plt.figure(figsize=(6 * 1.8, 3 * 1.8))

    nx.draw_networkx_nodes(G, pos, node_color='deepskyblue', node_size=950,
                           edgecolors='black', linewidths=2)
    labels = {cr.id: f"CR{cr.id}" for cr in network.crossroad_list}
    nx.draw_networkx_labels(G, pos, labels, font_size=13, font_color='navy', font_weight='bold')

    nx.draw_networkx_edges(G, pos, edge_color='grey', arrows=True,
                           arrowstyle='-|>', width=2.2, arrowsize=25,
                           connectionstyle='arc3,rad=0')

    edge_labels = {(s.begin.id, s.end.id): f"S{s.id}"
                   for s in network.street_list if s.begin and s.end}
    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_color="darkgreen", font_size=10, font_weight="bold")

    plt.gca().set_aspect('equal')
    plt.axis('off')
    plt.tight_layout(pad=1.5)
    plt.show()


# ============================================================
# Plots for a solved network (steady-state profiles and metrics)
# ============================================================
def plot_street_profiles(street: Street, b, n_points=200, show=True):
    """
    Plot density q(x), nonlocal impact W(x) and velocity v(x) for one street.
    """
    x, q_values, W, v = street.steady_state_profiles(b, n_points=n_points)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), sharex=True)
    axes[0].plot(x, q_values, color='C0')
    axes[0].set_title(f'S{street.id}: density $q(x)$')
    axes[0].set_ylabel(r'$q$')

    axes[1].plot(x, W, color='C1')
    axes[1].set_title(f'S{street.id}: nonlocal impact $W(x)$')
    axes[1].set_ylabel(r'$W$')

    axes[2].plot(x, v, color='C2')
    axes[2].set_title(f'S{street.id}: velocity $v(x)$')
    axes[2].set_ylabel(r'$v$')

    for ax in axes:
        ax.set_xlabel(r'$x$')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Street {street.id}: CR{street.begin.id} → CR{street.end.id}', y=1.02)
    fig.tight_layout()
    if show:
        plt.show()
    return fig, axes


def plot_network_density(network: Network, b, streets=None, street_colors=None, n_points=200, show=True, save_path=None):
    """Plot density q(x) for every (or selected) street in one figure."""
    return _plot_network_quantity(
        network, b, quantity='q', streets=streets, street_colors=street_colors,
        n_points=n_points, show=show, save_path=save_path,
    )


def plot_network_nonlocal_impact(network: Network, b, streets=None, street_colors=None, n_points=200, show=True, save_path=None):
    """Plot nonlocal impact W(x) for every (or selected) street in one figure."""
    return _plot_network_quantity(
        network, b, quantity='W', streets=streets, street_colors=street_colors,
        n_points=n_points, show=show, save_path=save_path,
    )


def plot_network_velocity(network: Network, b, streets=None, street_colors=None, n_points=200, show=True, save_path=None):
    """Plot velocity v(x) for every (or selected) street in one figure."""
    return _plot_network_quantity(
        network, b, quantity='v', streets=streets, street_colors=street_colors,
        n_points=n_points, show=show, save_path=save_path,
    )


def plot_network_profiles(network: Network, b, streets=None, street_colors=None, n_points=200, show=True, save_path=None):
    """
    Plot q, W and v for all (or selected) streets: one subplot row per quantity,
    one curve per street.

    street_colors: optional dict {street.id: color}, same as edge_colors in
    draw_network_grid_colored, so plot colors match the grid drawing.
    """
    if streets is None:
        streets = network.street_list

    fig, axes = plt.subplots(3, 1, figsize=(8, 9), sharex=False)
    quantity_meta = [
        ('q', r'density $q(x)$', r'$q$'),
        ('W', r'nonlocal impact $W(x)$', r'$W$'),
        ('v', r'velocity $v(x)$', r'$v$'),
    ]

    for street in streets:
        x, q_values, W, v = street.steady_state_profiles(b, n_points=n_points)
        data = {'q': q_values, 'W': W, 'v': v}
        color = None if street_colors is None else street_colors.get(street.id)
        for ax, (key, _title, _ylab) in zip(axes, quantity_meta):
            ax.plot(x, data[key], label=f'S{street.id}', color=color)

    for ax, (_key, title, ylab) in zip(axes, quantity_meta):
        ax.set_title(title)
        ax.set_ylabel(ylab)
        ax.set_xlabel(r'$x$')
        ax.grid(True, alpha=0.3)
        ax.legend(ncol=min(4, max(1, len(streets))), fontsize=8)

    fig.suptitle('Steady-state profiles on all streets', y=1.01)
    fig.tight_layout()
    _finalize_figure(fig, show=show, save_path=save_path)
    return fig, axes


def _plot_network_quantity(network: Network, b, quantity, streets=None, street_colors=None, n_points=200, show=True, save_path=None):
    if streets is None:
        streets = network.street_list

    titles = {
        'q': (r'density $q(x)$', r'$q$'),
        'W': (r'nonlocal impact $W(x)$', r'$W$'),
        'v': (r'velocity $v(x)$', r'$v$'),
    }
    if quantity not in titles:
        raise ValueError(f"quantity must be one of {list(titles)}, got {quantity!r}")

    title, ylab = titles[quantity]
    fig, ax = plt.subplots(figsize=(8, 4))

    for street in streets:
        x, q_values, W, v = street.steady_state_profiles(b, n_points=n_points)
        data = {'q': q_values, 'W': W, 'v': v}
        color = None if street_colors is None else street_colors.get(street.id)
        ax.plot(
            x, data[quantity],
            label=f'S{street.id} (CR{street.begin.id}→CR{street.end.id})',
            color=color,
        )

    ax.set_title(title)
    ax.set_xlabel(r'$x$')
    ax.set_ylabel(ylab)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _finalize_figure(fig, show=show, save_path=save_path)
    return fig, ax


def plot_street_travel_times(network: Network, b, street_colors=None, n_points=200, show=True, save_path=None):
    """Bar chart of steady-state travel time per street."""
    times = network.street_travel_times(b, n_points=n_points)
    ids = list(times.keys())
    vals = [times[i] for i in ids]
    if street_colors is None:
        colors = 'steelblue'
    else:
        colors = [street_colors.get(i, 'lightgrey') for i in ids]

    fig, ax = plt.subplots(figsize=(max(6, 0.5 * len(ids)), 4))
    ax.bar([f'S{i}' for i in ids], vals, color=colors, edgecolor='black')
    ax.set_ylabel('travel time')
    ax.set_title('Travel time per street')
    ax.grid(True, axis='y', alpha=0.3)
    fig.tight_layout()
    _finalize_figure(fig, show=show, save_path=save_path)
    return fig, ax


def draw_network_colored_by_metric(
    network: Network,
    b,
    metric='buffer',
    pos=None,
    n_points=200,
    show=True,
):
    """
    Draw the network graph with edges colored by a scalar metric.

    metric:
      - 'buffer':       b[street.id]
      - 'travel_time':  DALETH_s
      - 'flux':         phi_out = C * b[street.id]
    pos: optional {cr.id: (x, y)}; otherwise spring_layout is used.
    """
    G = _build_graph(network)
    if pos is None:
        pos = nx.spring_layout(G, seed=0)

    values = []
    edges = []
    for street in network.street_list:
        if street.begin is None or street.end is None:
            continue
        edges.append((street.begin.id, street.end.id))
        if metric == 'buffer':
            values.append(float(b[street.get_global_id()]))
        elif metric == 'travel_time':
            values.append(float(street.travel_time(b, n_points=n_points)))
        elif metric == 'flux':
            values.append(float(network.phi_out_street_func(street, b)))
        else:
            raise ValueError("metric must be 'buffer', 'travel_time' or 'flux'")

    values = np.asarray(values, dtype=float)
    fig, ax = plt.subplots(figsize=(8, 6))

    nx.draw_networkx_nodes(
        G, pos, ax=ax, node_color='lightgrey', node_size=700,
        edgecolors='black', linewidths=1.5,
    )
    labels = {cr.id: f'CR{cr.id}' for cr in network.crossroad_list}
    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=10)

    if len(values) == 0:
        raise ValueError('no streets to draw')

    vmin, vmax = float(np.min(values)), float(np.max(values))
    if vmin == vmax:
        vmax = vmin + 1e-12

    cmap = plt.cm.viridis
    # draw edges one by one for individual colors
    for (u, w), val in zip(edges, values):
        color = cmap((val - vmin) / (vmax - vmin))
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, w)], ax=ax, edge_color=[color],
            width=3.0, arrows=True, arrowstyle='-|>', arrowsize=18,
            connectionstyle='arc3,rad=0.05',
        )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric)

    ax.set_title(f'Network colored by {metric}')
    ax.set_aspect('equal')
    ax.axis('off')
    fig.tight_layout()
    if show:
        plt.show()
    return fig, ax


def test_func_01():
    params = {
        'b': 1,
        'lambda': 1.0,
        'eta': 1.0,
        'L': 1.0,
        'q_max': 1.0  # ,
        # 'rho_l': 1.0,
        # 'rho_r': 1e-6
    }

    s1 = Street(params)
    s1.print_info()

    myFirstCR = Crossroad()
    myLonelyCR = Crossroad()
    # myFirstOrigin = Crossroad()
    # myFirstDestination = Crossroad()

    # myFirstCR.add_predecessor(myFirstOrigin, s1, cascade=True)
    # myFirstCR.add_successor(myFirstDestination, s2, cascade=True)

    myFirstCR.create_and_add_predecessor(s1, cascade=True)
    myFirstCR.create_and_add_successor(s1, cascade=True)

    print("=" * 80)
    myFirstCR.streets_incoming[0].print_info()
    myFirstCR.streets_outgoing[0].print_info()

    print("=" * 80)
    myFirstCR.print_info()
    myFirstCR.predecessors[0].print_info()
    # myFirstOrigin.print_info()
    myFirstCR.successors[0].print_info()
    # myFirstDestination.print_info()
    myLonelyCR.print_info()
    myLonelyCR.create_and_add_predecessor(s1)
    myLonelyCR.print_info()
    print("=" * 80)

    myLonelyCR.streets_incoming[0].print_info()


def test_func_02():
    rows = 3
    cols = 3
    build_rowsxcols_network_auto(rows, cols)
    draw_network_grid(rows=rows, cols=cols)


def test_find_index_in_list():
    params = {
        'lambda': 1.0,
        'eta': 1.0,
        'L': 1.0,
        'q_max': 1.0  # ,
        # 'rho_l': 1.0,
        # 'rho_r': 1e-6
    }
    s0 = Street(params)   
    s1 = Street(params)   
    s2 = Street(params)   
    s3 = Street(params)   

    _l = [s0, s1, s2]

    s = s3
    if s in _l:
        print(_l.index(s))


def test_network_build():
    cols = 3
    m = 2
    network = build_network_with_only_1xm_crossroads(cols, m)

    network.solve()


def single_line(): 

    params = utils.default_params
    s0 = Street(params, constant_in_f=10)

    cr0 = Crossroad()
    cr = cr0
    for _ in range(3):
        cr = cr.create_and_add_successor(s0)

    network = Network(Crossroad.crossroad_list, Street.street_list)

    b = network.solve()

    print("b:", b)


def test_calculate_rho_l_and_W_0():
    params = utils.default_params

    s_template = Street(params, constant_in_f=10)
    # print(s_template)
    # s_template.print_info()

    cr0 = Crossroad()

    cr0.create_and_add_successor(s_template)

    s = cr0.streets_outgoing[0]

    b = np.ones(1)

    s.calculate_rho_l_and_W_0(b)


def test_jerome_rho_l_function_AND_calculate_rho_l_and_W_0(): 
    params = utils.default_params
    s_template = Street(params, constant_in_f=10)

    cr0 = Crossroad()

    cr1 = cr0.create_and_add_successor(s_template)
    cr2 = cr1.create_and_add_successor(s_template)
    cr2.create_and_add_successor(s_template)

    s = cr1.streets_outgoing[0]

    b = 0.04400769 * np.ones(3)

    # network has to be created for probs to work
    Network(Crossroad.crossroad_list, Street.street_list)

    # s.calculate_rho_l_and_W_0(b)
    try: 
        s.calculate_rho_l_and_W_0(b)
    except Exception as e:
        print("calculate_rho_l_and_W_0 failed: ", e)

    try: 
        rho_l = s.jerome_rho_l_function(b)
        print("rho_l from jerome:", rho_l)
        print("W_0 from jerome:", s.W_0_expr(b, rho_l))
    except Exception as e: 
        print("jerome_rho_l_function failed: ", e)


def solve_braexes_network():
    network_1 = build_braess_network(False)
    # draw_network_braess(network_1)

    network_2 = build_braess_network(True)
    # draw_network_braess(network_2)

    assert network_1.street_list[0].begin.is_origin
    assert network_1.street_list[5].end.is_destination
    assert network_1.street_list[-1] == network_1.street_list[5]

    assert network_2.street_list[0].begin.is_origin
    assert network_2.street_list[5].end.is_destination

    # network_1.print_street_list()
    # network_2.print_street_list()

    b1 = network_1.solve()
    b2 = network_2.solve()

    # Paths without the extra Braess edge:
    #   upper: S0 -> S1 -> S4 -> S5
    #   lower: S0 -> S2 -> S3 -> S5
    s1 = network_1.street_list
    path_upper_1 = [s1[0], s1[1], s1[4], s1[5]]
    path_lower_1 = [s1[0], s1[2], s1[3], s1[5]]

    # print("Network 1 street travel times:", network_1.street_travel_times(b1))
    print("Network 1 path upper:", network_1.path_travel_time(path_upper_1, b1))
    print("Network 1 path lower:", network_1.path_travel_time(path_lower_1, b1))

    s2 = network_2.street_list
    path_upper_2 = [s2[0], s2[1], s2[4], s2[5]]
    path_lower_2 = [s2[0], s2[2], s2[3], s2[5]]
    # optional middle path: S0 -> S1 -> S6 (cr2->cr3) -> S3 -> S5
    path_middle_2 = [s2[0], s2[1], s2[6], s2[3], s2[5]]

    print()
    # print("Network 2 street travel times:", network_2.street_travel_times(b2))
    print("Network 2 path upper:", network_2.path_travel_time(path_upper_2, b2))
    print("Network 2 path lower:", network_2.path_travel_time(path_lower_2, b2))
    print("Network 2 path middle:", network_2.path_travel_time(path_middle_2, b2))


def solve_line_like_braess():
    s0 = Street(utils.default_params)
    s_additional = Street(utils.params_for_short_street)

    cr = Crossroad()
    for i in range(5):
        if i == 2:
            cr = cr.create_and_add_successor(s_additional)
        else:
            cr = cr.create_and_add_successor(s0)

    network = Network(Crossroad.crossroad_list, Street.street_list)

    b = network.solve()

    # draw_network_grid(6, 1, network)
    # draw_network_braess(network)

    # network.print_street_list()

    # print("Network street travel times:", network.street_travel_times(b))
    print("Total travel time:", network.path_travel_time(network.street_list, b))


def showcase_network():
    cols, rows = 3, 3

    network = build_rowsxcols_network_auto(cols, rows)

    # Hardcode which streets share a color (edit this dict as needed).
    # Streets with the same color string are drawn identically.

    b = network.solve()
    for idx, value in enumerate(b):
        print(f"b_star[{idx}]: {value}")

    # Colors used for automatically detected groups (identical b => same group).
    # Edit this list; groups are assigned in order of sorted unique b values.
    # group_colors = ['C0', 'C1', 'C2', 'C3', 'C4', 'C5']
    group_colors = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

    unique_b = np.unique(np.asarray(b, dtype=float))
    if len(unique_b) > len(group_colors):
        raise ValueError(
            f"Need at least {len(unique_b)} group_colors, got {len(group_colors)}"
        )
    b_value_to_color = {
        val: group_colors[i] for i, val in enumerate(unique_b)
    }
    edge_colors = {
        street.id: b_value_to_color[b[street.get_global_id()]]
        for street in network.street_list
    }

    # Save plots to MOSI/images (do not open interactive windows).
    draw_network_grid(
        cols, rows, network,
        show=False,
        save_path=IMAGES_DIR / "3x3_grid_one_color.png",
    )
    draw_network_grid_colored(
        cols, rows, network, edge_colors=edge_colors,
        show=False,
        save_path=IMAGES_DIR / "3x3_grid.png",
    )

    # In the 3x3 grid many streets are symmetric mirrors; plot a representative subset.
    street_ids = {0, 2, 4, 9}
    streets = [s for s in network.street_list if s.id in street_ids]

    # Same edge_colors as the grid, so curves match street colors in the drawing.
    plot_network_density(
        network, b, streets=streets, street_colors=edge_colors,
        show=False, save_path=IMAGES_DIR / "3x3_density.png",
    )
    plot_network_nonlocal_impact(
        network, b, streets=streets, street_colors=edge_colors,
        show=False, save_path=IMAGES_DIR / "3x3_nonlocimp.png",
    )
    plot_network_velocity(
        network, b, streets=streets, street_colors=edge_colors,
        show=False, save_path=IMAGES_DIR / "3x3_velocity.png",
    )
    plot_street_travel_times(
        network, b, street_colors=edge_colors,
        show=False, save_path=IMAGES_DIR / "3x3_traveltime.png",
    )


if __name__ == "__main__":
    # test_func_01()
    # test_func_02()
    # test_func_03()
    # test_network_build()

    # single_line()

    # solve_braexes_network()
    # print()
    # solve_line_like_braess()
    # test_clear_all()

    showcase_network()

    pass
