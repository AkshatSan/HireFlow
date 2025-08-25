import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Even Number Challenge", layout="centered")

st.title("Even Number Challenge 🧮")

# Input for even number
num = st.number_input("Enter an even number:", step=1)

if st.button("Submit"):
    if num % 2 == 0:
        st.success(f"✅ Correct! {num} is even.")
    else:
        st.error(f"❌ {num} is odd. Try again.")

# Inject JavaScript to detect tab switch / blur event
# --- JS: ONLY tab switch/minimize via visibilitychange ---
import streamlit.components.v1 as components
components.html(
    """
    <script>
      (function () {
        if (window.__cheatGuardInstalled) return;
        window.__cheatGuardInstalled = true;

        let lastHiddenAt = 0;
        let lastAlertAt = 0;

        function showAlert(reason) {
          const now = Date.now();
          // prevent double alerts from rapid duplicate events
          if (now - lastAlertAt < 800) return;
          lastAlertAt = now;
          try { alert("⚠️ Don't cheat! Stay on this page. (" + reason + ")"); } catch (e) {}
        }

        function flagCheatAndRerun() {
          try {
            const url = new URL(window.parent.location.href);
            const params = new URLSearchParams(url.hash ? url.hash.substring(1) : "");
            params.set("cheat", "1");                 // lock the form on the Streamlit side
            params.set("_ts", Date.now().toString()); // force rerun
            url.hash = params.toString();
            window.parent.location.replace(url.toString());
          } catch (e) {
            try {
              const url2 = new URL(window.location.href);
              const params2 = new URLSearchParams(url2.hash ? url2.hash.substring(1) : "");
              params2.set("cheat", "1");
              params2.set("_ts", Date.now().toString());
              url2.hash = params2.toString();
              window.location.replace(url2.toString());
            } catch (e2) {}
          }
        }

        // ONLY handler: fires when page becomes hidden/visible (tab switch/minimize/restore)
        document.addEventListener("visibilitychange", () => {
          if (document.hidden) {
            lastHiddenAt = Date.now();
            showAlert("tab switched / hidden");
            flagCheatAndRerun();  // lock form after first switch
          } else {
            // Some browsers suppress alerts while backgrounded; alert again on return
            if (Date.now() - lastHiddenAt < 60000) {
              showAlert("returned after tab switch");
            }
          }
        });
      })();
    </script>
    """,
    height=0,
)

