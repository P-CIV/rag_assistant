document.addEventListener("DOMContentLoaded", async () => {

    const zoneMessages  = document.getElementById("messages-chat");
    const champSaisie   = document.getElementById("champ-utilisateur");
    const boutonEnvoyer = document.getElementById("bouton-envoyer");
    const barreStatut   = document.getElementById("barre-statut");

    const API_URL        = "https://rag-assistant-7l82.onrender.com";
    const CLE_SESSION    = "kova_session_id";
    const CLE_HISTORIQUE = "kova_historique";
    const MAX_HISTORIQUE = 100;
    const MAX_TENTATIVES = 4;
    const DELAI_RETRY_MS = 12000;

    let sessionId    = sessionStorage.getItem(CLE_SESSION) || null;
    let sessionPrete = !!sessionId;

    // Active le champ dès le départ l'utilisateur peut écrire pendant le démarrage.
    definirEtatSaisie(true);

    // Utilitaires statut

    function afficherStatut(texte, type = "attente") {
        const texteStatut = document.getElementById("texte-statut-serveur");
        if (texteStatut) texteStatut.textContent = texte;
        if (barreStatut) barreStatut.className = `barre-statut barre-statut--${type}`;
    }

    function masquerStatut() {
        if (barreStatut) barreStatut.className = "barre-statut barre-statut--cachee";
    }

    // Initialisation session

    async function initialiserSession(tentative = 1) {
        afficherStatut(
            tentative === 1
                ? "Connexion au serveur..."
                : `Serveur en démarrage — tentative ${tentative} / ${MAX_TENTATIVES}...`,
            "attente"
        );

        try {
            const res = await fetch(`${API_URL}/session/nouvelle`, {
                method: "POST",
                signal: AbortSignal.timeout(20000),
            });
            const data = await res.json();
            sessionId    = data.session_id;
            sessionPrete = true;
            sessionStorage.setItem(CLE_SESSION, sessionId);
            afficherStatut("Connecté", "succes");
            setTimeout(masquerStatut, 2000);
        } catch {
            if (tentative < MAX_TENTATIVES) {
                let secondes = DELAI_RETRY_MS / 1000;
                const compte = setInterval(() => {
                    secondes--;
                    if (secondes > 0) {
                        afficherStatut(
                            `Serveur en démarrage — nouvelle tentative dans ${secondes} s...`,
                            "attente"
                        );
                    } else {
                        clearInterval(compte);
                    }
                }, 1000);
                await new Promise(r => setTimeout(r, DELAI_RETRY_MS));
                clearInterval(compte);
                return initialiserSession(tentative + 1);
            } else {
                afficherStatut(
                    "Serveur inaccessible. Votre message sera envoyé dès la reconnexion.",
                    "erreur"
                );
                sessionPrete = false;
            }
        }
    }

    if (!sessionId) {
        initialiserSession(); // lancé en arrière-plan, ne bloque pas le champ
    } else {
        masquerStatut();
    }

    // Helpers UI

    function heureActuelle() {
        const d = new Date();
        return `${d.getHours()}:${String(d.getMinutes()).padStart(2, "0")}`;
    }

    function definirEtatSaisie(actif) {
        if (champSaisie)   champSaisie.disabled   = !actif;
        if (boutonEnvoyer) boutonEnvoyer.disabled = !actif;
        if (actif && champSaisie) champSaisie.focus();
    }

    function scrollerBas() {
        if (zoneMessages) zoneMessages.scrollTop = zoneMessages.scrollHeight;
    }

    function detecterIcone(titre) {
        const t = titre.toLowerCase();
        if (t.includes("maintenance") || t.includes("entretien")) return "fa-tools";
        if (t.includes("inspection")  || t.includes("audit"))     return "fa-search";
        if (t.includes("formation"))                               return "fa-graduation-cap";
        if (t.includes("contrat")     || t.includes("offre"))     return "fa-file-contract";
        if (t.includes("hydraulique") || t.includes("lectrique")) return "fa-bolt";
        return "fa-cog";
    }

    function formatInline(t) {
        return t
            .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
            .replace(/\*([^*]+)\*/g,   "<em>$1</em>");
    }

    // Supprime l'extension .pdf
    function supprimerExtension(fichier) {
        return fichier.replace(/\.pdf$/i, "");
    }

    function abrevierNom(fichier) {
        const nom = supprimerExtension(fichier);
        if (nom.length <= 28) return nom;
        return nom.slice(0, 18) + "..." + nom.slice(-7);
    }

    // formaterCitation retire l'extension .pdf avant l'affichage inline
    function formaterCitation(texteSource) {
        const regex = /([^,\s*]+\.(?:pdf|PDF))[,\s]*(pages?\s*[\d,\set]+)?/gi;
        const resultats = [];
        let match;
        while ((match = regex.exec(texteSource)) !== null) {
            resultats.push({ fichier: match[1], pages: match[2] ? match[2].trim() : null });
        }
        if (resultats.length === 0) return texteSource;
        return resultats.map(r => {
            const nom = abrevierNom(r.fichier);
            return r.pages ? `${nom} — ${r.pages}` : nom;
        }).join("  |  ");
    }

    function formaterTexte(texte) {
        const lignes = texte.split("\n");
        let html = "";
        let section = null;

        function fermerSection() {
            if (section !== null) {
                html += `<div class="bloc-service">${section}</div>`;
                section = null;
            }
        }

        lignes.forEach(ligne => {
            const t = ligne.trim();
            if (!t) return;

            const matchTitre  = t.match(/^\*\*(.+)\*\*$/);
            const matchSource = t.match(/^\*\((?:Source\s*:?\s*)?(.+)\)\*$/i);

            if (matchTitre) {
                fermerSection();
                const icone = detecterIcone(matchTitre[1]);
                section = `<div class="titre-bloc"><i class="fas ${icone}"></i>${matchTitre[1]}</div>`;
            } else if (matchSource) {
                const contenu = formaterCitation(matchSource[1]);
                if (!contenu) return;
                const citationHtml = `<div class="citation-source"><i class="fas fa-file-alt"></i>${contenu}</div>`;
                if (section !== null) section += citationHtml;
                else html += citationHtml;
            } else {
                const formatted = formatInline(t);
                if (section !== null) section += `<p>${formatted}</p>`;
                else html += `<p class="intro-p">${formatted}</p>`;
            }
        });

        fermerSection();
        return html;
    }

    function ajouterMessage(texte, estUtilisateur = false, sources = []) {
        const message = document.createElement("div");
        message.className = `message ${estUtilisateur ? "message-utilisateur" : "message-bot"}`;

        const contenu = document.createElement("div");
        contenu.className = "contenu-message";
        contenu.innerHTML = estUtilisateur ? `<p>${texte}</p>` : formaterTexte(texte);
        message.appendChild(contenu);

        if (!estUtilisateur && sources.length > 0) {
            message.appendChild(creerBlocSources(sources));
        }

        const heure = document.createElement("div");
        heure.className   = "heure-message";
        heure.textContent = heureActuelle();
        message.appendChild(heure);

        zoneMessages.appendChild(message);
        scrollerBas();
    }

    // Normalise un nom de fichier pour la comparaison : sans extension, sans "page X", en minuscules
    function normaliserNom(fichier) {
        return fichier
            .replace(/,?\s*pages?\s*[\d,\set]+/gi, "")
            .replace(/\.pdf$/i, "")
            .trim()
            .toLowerCase();
    }

    // Extrait les noms de fichiers PDF cités dans le texte (avec ou sans astérisques autour)
    function extraireFichiersCites(texte) {
        const cites = new Set();
        const re = /([^,\s*(]+\.(?:pdf|PDF))/gi;
        let hit;
        while ((hit = re.exec(texte)) !== null) {
            cites.add(normaliserNom(hit[1]));
        }
        return cites;
    }

    // Ne montrer dans le panneau sources que les fichiers réellement cités.
    // Si aucune citation n'est détectée mais que des sources existent, on les affiche toutes
    // (cas où le LLM mentionne les sources sans format PDF explicite dans le corps du texte).
    function afficherMessageBot(texte, sources = []) {
        const cites = extraireFichiersCites(texte);

        if (cites.size === 0) {
            ajouterMessage(texte, false, sources);
            return;
        }

        const sourcesFiltrees = sources.filter(({ fichier }) =>
            cites.has(normaliserNom(fichier))
        );

        // Si le filtre est trop strict et exclut tout, on affiche toutes les sources
        ajouterMessage(texte, false, sourcesFiltrees.length > 0 ? sourcesFiltrees : sources);
    }

    // Typing indicator

    let timerColdStart = null;

    function afficherTyping() {
        const el = document.createElement("div");
        el.id        = "typing-indicator";
        el.className = "message message-bot";
        el.innerHTML = `<div class="contenu-message typing"><span></span><span></span><span></span></div>`;
        zoneMessages.appendChild(el);
        scrollerBas();
        timerColdStart = null;
    }

    function supprimerTyping() {
        clearTimeout(timerColdStart);
        document.getElementById("typing-indicator")?.remove();
    }

    // Bloc sources

    function creerBlocSources(sources) {
        // Dédupliquer par nom (sans "page X", avec extension .pdf conservée pour l'affichage)
        const map = new Map();
        sources.forEach(({ fichier, score }) => {
            const nom = fichier
                .replace(/,?\s*page\s*\d+/gi, "")
                .trim();
            if (!map.has(nom) || score > map.get(nom)) map.set(nom, score);
        });
        const sourcesUniques = Array.from(map.entries());

        const bloc = document.createElement("div");
        bloc.className = "sources-message";

        const btn = document.createElement("button");
        btn.className = "toggle-sources";
        btn.innerHTML = `<i class="fas fa-file-alt"></i>${sourcesUniques.length} source${sourcesUniques.length > 1 ? "s" : ""}`;

        const liste = document.createElement("ul");
        liste.className = "liste-sources cachee";

        sourcesUniques.forEach(([fichier, score]) => {
            const classBadge = score >= 70 ? "score-vert" : score >= 40 ? "score-jaune" : "score-rouge";
            const label      = score >= 70 ? "Très pertinent" : score >= 40 ? "Pertinent" : "Peu pertinent";
            const li         = document.createElement("li");
            const nomAffiche = supprimerExtension(fichier);
            li.innerHTML     = `
                <i class="fas fa-file-pdf source-icone"></i>
                <span class="source-nom">${nomAffiche}</span>
                <span class="score-badge ${classBadge}">${score} % - ${label}</span>`;
            liste.appendChild(li);
        });

        btn.addEventListener("click", () => {
            liste.classList.toggle("cachee");
            btn.classList.toggle("active");
        });

        bloc.appendChild(btn);
        bloc.appendChild(liste);
        return bloc;
    }

    // API

    async function appellerAPI(messageUtilisateur) {
        const res = await fetch(`${API_URL}/chat`, {
            method:  "POST",
            headers: { "Content-Type": "application/json" },
            body:    JSON.stringify({ session_id: sessionId, message: messageUtilisateur }),
            signal:  AbortSignal.timeout(90000),
        });
        if (!res.ok) {
            const err = await res.json().catch(() => ({}));
            throw new Error(err.detail || `Erreur serveur ${res.status}`);
        }
        return res.json();
    }

    function sauvegarderEchange(question, reponse) {
        const historique = JSON.parse(localStorage.getItem(CLE_HISTORIQUE) || "[]");
        historique.push({ date: new Date().toISOString(), question, reponse });
        if (historique.length > MAX_HISTORIQUE) historique.splice(0, historique.length - MAX_HISTORIQUE);
        localStorage.setItem(CLE_HISTORIQUE, JSON.stringify(historique));
    }

    // Envoi message

    async function envoyerMessage() {
        const texte = champSaisie.value.trim();
        if (!texte) return;

        // Si la session n'est pas prête, on attend qu'elle s'initialise.
        if (!sessionId) {
            afficherStatut("Connexion en cours…", "attente");
            await initialiserSession();
            if (!sessionId) {
                afficherMessageBot(
                    "Impossible de joindre le serveur. Vérifiez votre connexion et réessayez."
                );
                return;
            }
        }

        ajouterMessage(texte, true);
        champSaisie.value = "";
        definirEtatSaisie(false);
        afficherTyping();

        try {
            const data = await appellerAPI(texte);
            supprimerTyping();
            afficherMessageBot(data.reponse, data.sources || []);
            sauvegarderEchange(texte, data.reponse);
        } catch (erreur) {
            supprimerTyping();
            const msg = erreur.name === "TimeoutError" ? null : `Erreur : ${erreur.message}`;
            if (!msg) { definirEtatSaisie(true); return; }
            afficherMessageBot(msg);
            console.error("[KOVA] Erreur API :", erreur);
        } finally {
            definirEtatSaisie(true);
        }
    }

    boutonEnvoyer.addEventListener("click", envoyerMessage);

    champSaisie.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            envoyerMessage();
        }
    });

    window.addEventListener("beforeunload", () => {
        if (sessionId) {
            navigator.sendBeacon(`${API_URL}/session/${sessionId}`, "{}");
            sessionStorage.removeItem(CLE_SESSION);
        }
    });

});