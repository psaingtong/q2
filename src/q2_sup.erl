-module(q2_sup).
-behaviour(supervisor).

-export([start_link/0]).

-export([init/1]).

-define(SERVER, ?MODULE).

start_link() ->
    supervisor:start_link({local, ?SERVER}, ?MODULE, []).

%% sup_flags() = #{strategy => strategy(),         % optional
%%                 intensity => non_neg_integer(), % optional
%%                 period => pos_integer()}        % optional
%% child_spec() = #{id => child_id(),       % mandatory
%%                  start => mfargs(),      % mandatory
%%                  restart => restart(),   % optional
%%                  shutdown => shutdown(), % optional
%%                  type => worker(),       % optional
%%                  modules => modules()}   % optional
init([]) ->
    %{ok, PktFwdOpts} = application:get_env(q2, packet_forwarder_listen),
    SupFlags = #{
        strategy => one_for_one,
        intensity => 2,
        period => 10
    },
    ChildSpecs = [
        ],
    {ok, {SupFlags, ChildSpecs}}.

%% internal functions